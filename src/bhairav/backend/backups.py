"""PostgreSQL logical backups + live DB metrics (Phase 9 M3).

A pure-Python pg_dump-style logical backup (no pg_dump binary): one
transaction reads every public table's schema (exact types via
format_type) and rows (BYTEA -> base64, JSONB -> nested JSON), gzips the
manifest to ``bhairav_<utc>.backup.json.gz`` and applies retention. The
matching restore recreates tables and rows in any PostgreSQL target, so a
backup made on the cluster can be restored to a scratch database for
verification or a fresh cluster after disaster. ``pg_metrics`` feeds the
health dashboard / Prometheus with live DB size and row counts.
"""
from __future__ import annotations

import base64
import gzip
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

BACKUP_FORMAT = "bhairav-logical-backup"
BACKUP_VERSION = 1
BACKUP_RE = re.compile(r"^bhairav_(\d{8})_(\d{6})\.backup\.json\.gz$")


def _driver():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "PostgreSQL backups require psycopg 3. Install it with: "
            'pip install "psycopg[binary]==3.3.4"') from exc
    return psycopg


def _connect(db_url):
    conn = _driver().connect(db_url, connect_timeout=5)
    try:
        # register jsonb <-> dict adapters (psycopg3 needs this explicitly)
        from psycopg.types.json import register_default_adapters
        register_default_adapters(conn)
    except ImportError:  # pragma: no cover - older psycopg builds
        pass
    return conn


def _encode(value):
    """JSON-safe value encoding (bytes -> base64 marker, jsonb -> $json)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, memoryview)):
        return {"$b64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (dict, list)):
        return {"$json": value}
    if hasattr(value, "isoformat"):  # datetime/date/timestamp
        return value.isoformat()
    return str(value)


def _decode(value):
    if isinstance(value, dict) and set(value) == {"$b64"}:
        return base64.b64decode(value["$b64"])
    if isinstance(value, dict) and set(value) == {"$json"}:
        return value["$json"]
    return value


def _schema(conn, tables: list[str] | None = None) -> list[dict]:
    """Column DDL per public table (exact types via format_type)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.relname, a.attname,
               format_type(a.atttypid, a.atttypmod) AS typ,
               a.attnotnull,
               pg_get_expr(ad.adbin, ad.adrelid) AS default
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
          JOIN pg_attribute a ON a.attrelid = c.oid
          LEFT JOIN pg_attrdef ad ON ad.adrelid = c.oid AND ad.adnum = a.attnum
         WHERE n.nspname = 'public' AND c.relkind = 'r'
           AND a.attnum > 0 AND NOT a.attisdropped
         ORDER BY c.relname, a.attnum
        """)
    by_table: dict[str, list] = {}
    for table, col, typ, notnull, default in cur.fetchall():
        if tables and table not in tables:
            continue
        by_table.setdefault(table, []).append({
            "name": col, "type": typ,
            "not_null": bool(notnull),
            "default": default,
        })
    return [{"name": t, "columns": cols} for t, cols in sorted(by_table.items())]


def _dump_rows(conn, table: dict) -> list:
    cols = ", ".join(f'"{c["name"]}"' for c in table["columns"])
    cur = conn.cursor()
    cur.execute(f'SELECT {cols} FROM public."{table["name"]}"')
    return [[_encode(v) for v in row] for row in cur.fetchall()]


def dump(db_url: str, out_dir: str | Path, retention: int = 14,
         tables: list[str] | None = None) -> dict:
    """Full logical backup of the public schema into a gzip'd JSON manifest."""
    conn = _connect(db_url)
    try:
        with conn:
            schema = _schema(conn, tables)
            payload = {
                "format": BACKUP_FORMAT, "version": BACKUP_VERSION,
                "created_at": time.time(),
                "server_version": conn.info.server_version,
                "tables": [{"name": t["name"], "columns": t["columns"],
                            "rows": _dump_rows(conn, t)} for t in schema],
            }
    finally:
        conn.close()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"bhairav_{stamp}.backup.json.gz"
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    path.write_bytes(gzip.compress(raw, compresslevel=6))
    removed = prune(out_dir, retention)
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "created_at": payload["created_at"], "tables": len(payload["tables"]),
            "pruned": removed}


def _parse_stamp(name: str) -> float:
    """Epoch seconds from a backup filename; 0 when the name is malformed."""
    m = BACKUP_RE.match(name)
    if not m:
        return 0.0
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") \
            .replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def list_backups(out_dir: str | Path) -> list[dict]:
    """[{name, size_bytes, created_at, age_sec}], newest first."""
    out = Path(out_dir)
    rows = []
    if out.is_dir():
        for p in sorted(out.glob("bhairav_*.backup.json.gz")):
            created = _parse_stamp(p.name)
            rows.append({"name": p.name, "size_bytes": p.stat().st_size,
                         "created_at": created or p.stat().st_mtime,
                         "age_sec": round(time.time() - (created or p.stat().st_mtime), 1)})
    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return rows


def prune(out_dir: str | Path, retention: int) -> list[str]:
    """Keep the newest `retention` backups; return the removed filenames."""
    rows = list_backups(out_dir)
    removed = []
    for row in rows[retention:]:
        try:
            (Path(out_dir) / row["name"]).unlink()
            removed.append(row["name"])
        except OSError:
            pass
    return removed


def verify(path: str | Path) -> dict:
    """Validate a backup file: gzip + format + per-table shape."""
    try:
        raw = gzip.decompress(Path(path).read_bytes())
        data = json.loads(raw.decode("utf-8"))
        ok = (data.get("format") == BACKUP_FORMAT
              and data.get("version") == BACKUP_VERSION
              and isinstance(data.get("tables"), list))
        tables = []
        for t in data.get("tables", []):
            valid = (isinstance(t.get("columns"), list)
                     and isinstance(t.get("rows"), list))
            tables.append({"name": t.get("name"), "columns": len(t.get("columns", [])),
                           "rows": len(t.get("rows", [])), "valid": bool(valid)})
            ok = ok and valid
        return {"ok": ok, "tables": tables, "created_at": data.get("created_at"),
                "error": None}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "tables": [], "created_at": None, "error": str(exc)}


def _table_ddl(table: dict) -> str:
    cols = []
    for c in table["columns"]:
        ddl = f'"{c["name"]}" {c["type"]}'
        if c.get("not_null"):
            ddl += " NOT NULL"
        if c.get("default"):
            ddl += f" DEFAULT {c['default']}"
        cols.append(ddl)
    return f'CREATE TABLE public."{table["name"]}" ({", ".join(cols)})'


def restore(db_url: str, path: str | Path, wipe: bool = False,
            tables: list[str] | None = None) -> dict:
    """Restore a backup: recreate tables (optionally wiping) and insert rows.

    Returns {"tables": [{name, rows}], "total_rows"}. Existing tables are
    truncated (wipe=False) or dropped + recreated (wipe=True).
    """
    data = json.loads(gzip.decompress(Path(path).read_bytes()).decode("utf-8"))
    if data.get("format") != BACKUP_FORMAT:
        raise ValueError(f"not a BHAIRAV backup: {path}")
    conn = _connect(db_url)
    restored = []
    total = 0
    try:
        with conn:
            cur = conn.cursor()
            for table in data["tables"]:
                name = table["name"]
                if tables and name not in tables:
                    continue
                if wipe:
                    cur.execute(f'DROP TABLE IF EXISTS public."{name}" CASCADE')
                    cur.execute(_table_ddl(table))
                else:
                    cur.execute(_table_ddl(table).replace("CREATE TABLE",
                                                          "CREATE TABLE IF NOT EXISTS", 1))
                    cur.execute(f'TRUNCATE public."{name}"')
                cols = [c["name"] for c in table["columns"]]
                placeholders = ", ".join(["%s"] * len(cols))
                col_sql = ", ".join(f'"{c}"' for c in cols)
                stmt = (f'INSERT INTO public."{name}" ({col_sql}) '
                        f"VALUES ({placeholders})")
                # jsonb columns need an explicit Jsonb wrapper (pg_store does
                # the same); everything else maps 1:1 from the manifest.
                from psycopg.types.json import Jsonb
                preparers = []
                for col in table["columns"]:
                    if col["type"] == "jsonb":
                        preparers.append(lambda v, _J=Jsonb: _J(v)
                                         if isinstance(v, (dict, list)) else v)
                    else:
                        preparers.append(lambda v: v)
                rows = [[fn(_decode(v))
                         for fn, v in zip(preparers, row)]
                        for row in table["rows"]]
                if rows:
                    cur.executemany(stmt, rows)
                restored.append({"name": name, "rows": len(rows)})
                total += len(rows)
    finally:
        conn.close()
    return {"tables": restored, "total_rows": total}


def pg_metrics(db_url: str) -> dict:
    """Live database telemetry for /api/status and /metrics."""
    try:
        conn = _connect(db_url)
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_database_size(current_database())")
        size = cur.fetchone()[0]
        cur.execute(
            "SELECT relname, n_live_tup FROM pg_stat_user_tables "
            "WHERE schemaname = 'public' ORDER BY relname")
        rows = {name: int(n) for name, n in cur.fetchall()}
        return {"reachable": True, "db_size_bytes": int(size), "table_rows": rows,
                "version": conn.info.server_version}
    finally:
        conn.close()


class BackupService:
    """API-facing facade over dump/list/verify/prune for a fixed directory.

    create_app() receives one of these (or None); the REST layer only talks
    to this protocol, so tests can inject a stub.
    """

    def __init__(self, db_url: str, out_dir: str | Path, retention: int = 14):
        self.db_url = db_url
        self.out_dir = Path(out_dir)
        self.retention = retention

    def list(self) -> list[dict]:
        return list_backups(self.out_dir)

    def create(self) -> dict:
        return dump(self.db_url, self.out_dir, retention=self.retention)

    def read(self, name: str) -> bytes | None:
        """Raw bytes of a backup by filename (validated name)."""
        if not BACKUP_RE.match(name or ""):
            return None
        path = self.out_dir / name
        try:
            return path.read_bytes()
        except OSError:
            return None

    def latest(self) -> dict | None:
        rows = self.list()
        return rows[0] if rows else None

    def verify(self, name: str) -> dict:
        if not BACKUP_RE.match(name or ""):
            return {"ok": False, "tables": [], "created_at": None,
                    "error": "invalid backup name"}
        return verify(self.out_dir / name)
