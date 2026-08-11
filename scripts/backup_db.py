"""BHAIRAV Phase 9 M3 - PostgreSQL logical backup CLI.

Pure-Python pg_dump-style backup (gzip'd JSON, no pg_dump binary) with
retention pruning and a post-write verify pass.

Usage:
  python scripts/backup_db.py --url postgresql://bhairav:pass@localhost:5432/bhairav
  python scripts/backup_db.py --dir backups --retention 14 --verify
  python scripts/backup_db.py --list   # show what exists (no backup taken)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description="BHAIRAV PostgreSQL backup (M3)")
    ap.add_argument("--url", default=None,
                    help="PostgreSQL URL (default: $BHAIRAV_DB_URL)")
    ap.add_argument("--dir", default="backups", help="backup directory")
    ap.add_argument("--retention", type=int, default=14,
                    help="keep the newest N backups")
    ap.add_argument("--verify", action="store_true",
                    help="verify the written file (gzip + format) before exiting")
    ap.add_argument("--list", action="store_true", help="list backups and exit")
    ap.add_argument("--json-out", default=None, dest="json_out",
                    help="write the result as JSON to this path")
    args = ap.parse_args()

    from bhairav.backend.backups import dump, list_backups

    if args.list:
        rows = list_backups(args.dir)
        for r in rows:
            print(f"{r['name']:40s} {r['size_bytes']:>10d} bytes  "
                  f"{r['age_sec']:>9.1f}s old")
        print(f"total: {len(rows)} backup(s) in {args.dir}")
        return 0

    db_url = args.url or os.environ.get("BHAIRAV_DB_URL")
    if not db_url:
        print("no database: pass --url or set BHAIRAV_DB_URL", file=sys.stderr)
        return 2

    result = dump(db_url, args.dir, retention=args.retention)
    if args.verify:
        from bhairav.backend.backups import verify as _verify
        v = _verify(result["path"])
        result["verify"] = v
        if not v["ok"]:
            print(f"VERIFY FAILED: {v.get('error')}", file=sys.stderr)
            return 1
    print(f"backup written: {result['path']} "
          f"({result['size_bytes']} bytes, {result['tables']} tables)")
    if result.get("pruned"):
        print(f"pruned old backups: {', '.join(result['pruned'])}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2),
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
