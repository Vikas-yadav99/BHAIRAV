"""Migrate JSONL data to PostgreSQL.

Usage:
    python scripts/migrate_to_pg.py --db-url postgresql://user:pass@localhost/bhairav
    python scripts/migrate_to_pg.py --db-url postgresql://user:pass@localhost/bhairav --dry-run

Migrates:
    output/incidents/incidents.jsonl  → incidents table
    output/incidents/officers.jsonl   → officers table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def migrate_jsonl_to_pg(db_url: str, data_dir: str = "output/incidents",
                         dry_run: bool = False):
    """Migrate JSONL files to PostgreSQL."""
    from bhairav.backend.pg_incidents import PgIncidentStore

    data_path = Path(data_dir)
    inc_file = data_path / "incidents.jsonl"
    off_file = data_path / "officers.jsonl"

    if not inc_file.exists() and not off_file.exists():
        print(f"No JSONL files found in {data_dir}")
        return

    if dry_run:
        print("DRY RUN — would migrate:")
        if inc_file.exists():
            count = sum(1 for line in inc_file.read_text().splitlines() if line.strip())
            print(f"  {inc_file}: {count} incidents")
        if off_file.exists():
            count = sum(1 for line in off_file.read_text().splitlines() if line.strip())
            print(f"  {off_file}: {count} officers")
        return

    print(f"Connecting to: {db_url[:30]}...")
    store = PgIncidentStore(db_url)

    # Migrate incidents
    if inc_file.exists():
        migrated = 0
        for line in inc_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                sql = """INSERT INTO incidents
                    (id, category, emergency_level, location_lat, location_lng,
                     location_name, description, reporter_phone, reporter_name,
                     status, source, created_at, updated_at, assigned_officers,
                     resolution_notes, crowd_reports, ai_verified, timeline)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO NOTHING"""
                with store._conn.cursor() as cur:
                    cur.execute(sql, (
                        d["id"], d["category"], d["emergency_level"],
                        d["location_lat"], d["location_lng"],
                        d["location_name"], d["description"],
                        d.get("reporter_phone", ""), d.get("reporter_name", ""),
                        d["status"], d["source"],
                        d["created_at"], d["updated_at"],
                        d.get("assigned_officers", []),
                        d.get("resolution_notes", ""),
                        d.get("crowd_reports", 1),
                        d.get("ai_verified", False),
                        d.get("timeline", []),
                    ))
                migrated += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Skipped line: {e}")
        print(f"Migrated {migrated} incidents")

    # Migrate officers
    if off_file.exists():
        migrated = 0
        for line in off_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                store.add_officer(
                    officer_id=d["id"], name=d["name"], role=d.get("role", "responder"),
                    phone=d.get("phone", ""), lat=d["location_lat"], lng=d["location_lng"],
                    specialty=d.get("specialty", []),
                )
                migrated += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Skipped line: {e}")
        print(f"Migrated {migrated} officers")

    # Verify
    print(f"\nVerification:")
    print(f"  Incidents in PG: {store.count_incidents()}")
    print(f"  Officers in PG:  {len(store.list_officers())}")

    store.close()
    print("Migration complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate BHAIRAV JSONL to PostgreSQL")
    parser.add_argument("--db-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument("--data-dir", default="output/incidents", help="JSONL data directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    args = parser.parse_args()

    migrate_jsonl_to_pg(args.db_url, args.data_dir, args.dry_run)
