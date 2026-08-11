"""Phase 9 M5 - static deploy consistency checks.

These cannot replace a real `docker compose up`, but they lock the
cross-references that DO break deployments in obvious ways: compose env
names vs what serve.py actually reads, the Grafana datasource uid the
dashboard JSON references, the backup CLI args the compose backup service
passes, and the Dockerfile's model build-context contract.
"""
import json
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "deploy/docker-compose.yml").read_text(
    encoding="utf-8"))
SERVE_SRC = (ROOT / "scripts/serve.py").read_text(encoding="utf-8")


def _env_names(service: str) -> set[str]:
    env = COMPOSE["services"][service]["environment"]
    return {k.split(":")[0].split("?")[0] for k in env}


def test_compose_env_vars_are_read_by_serve():
    """Every BHAIRAV_* env the compose app sets must be honored by serve.py
    (plus the 12-factor DATABASE_URL, which compose relies on)."""
    for name in _env_names("app"):
        if name.startswith("BHAIRAV_"):
            assert name in SERVE_SRC, f"{name} not referenced by serve.py"
    assert "DATABASE_URL" in SERVE_SRC


def test_grafana_dashboard_datasource_uid_resolves():
    ds = (ROOT / "deploy/grafana/provisioning/datasources/prometheus.yml")
    ds_cfg = yaml.safe_load(ds.read_text(encoding="utf-8"))
    uids = {d.get("uid") for d in ds_cfg["datasources"]}
    dash = json.loads((ROOT / "deploy/grafana/dashboards/bhairav.json")
                      .read_text(encoding="utf-8"))
    # only uids inside "datasource": {...} blocks count (the dashboard's own
    # "uid" field is its identity, not a datasource reference)
    used = set(re.findall(r'"datasource"\s*:\s*\{[^}]*?"uid"\s*:\s*"([^"]+)"',
                          json.dumps(dash)))
    assert used <= uids, f"dashboard uids {used} missing from datasources {uids}"


def test_backup_service_cli_args_exist():
    """The compose backup service runs backup_db.py with these flags."""
    src = (ROOT / "scripts/backup_db.py").read_text(encoding="utf-8")
    for flag in ("--url", "--dir", "--retention", "--verify", "--list"):
        assert flag in src, f"backup_db.py missing {flag}"


def test_dockerfile_build_context_has_models_dir():
    """Dockerfile COPYs models/; a fresh clone must have the dir (via
    .gitkeep) or `docker build` fails."""
    df = (ROOT / "deploy/Dockerfile").read_text(encoding="utf-8")
    assert "COPY models ./models" in df
    assert (ROOT / "models/.gitkeep").exists()


def test_compose_services_parse_and_network():
    assert "app" in COMPOSE["services"]
    assert COMPOSE["services"]["app"]["depends_on"]["db"]["condition"] == \
        "service_healthy"
    # backup + prometheus + grafana from Phase 9 M3
    for svc in ("backup", "prometheus", "grafana", "db", "nginx"):
        assert svc in COMPOSE["services"], svc
