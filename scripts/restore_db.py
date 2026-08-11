"""BHAIRAV Phase 9 M3 - PostgreSQL restore CLI.

Restores a backup written by scripts/backup_db.py into a target database.
Existing tables are truncated by default; pass --wipe to drop + recreate.

Usage:
  python scripts/restore_db.py --url postgresql://... --file backups/bhairav_...json.gz
  python scripts/restore_db.py --url postgresql://... --file backup.gz --wipe --dry-run
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
    ap = argparse.ArgumentParser(description="BHAIRAV PostgreSQL restore (M3)")
    ap.add_argument("--url", default=None,
                    help="PostgreSQL URL (default: $BHAIRAV_DB_URL)")
    ap.add_argument("--file", required=True, help="backup .json.gz file")
    ap.add_argument("--wipe", action="store_true",
                    help="DROP TABLE + recreate instead of truncating")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify the file only; do not touch the database")
    args = ap.parse_args()

    from bhairav.backend.backups import restore, verify

    if args.dry_run:
        v = verify(args.file)
        print(json.dumps(v, indent=2))
        return 0 if v["ok"] else 1

    db_url = args.url or os.environ.get("BHAIRAV_DB_URL")
    if not db_url:
        print("no database: pass --url or set BHAIRAV_DB_URL", file=sys.stderr)
        return 2

    result = restore(db_url, args.file, wipe=args.wipe)
    print(f"restored {result['total_rows']} rows across "
          f"{len(result['tables'])} tables")
    for t in result["tables"]:
        print(f"  {t['name']:20s} {t['rows']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
