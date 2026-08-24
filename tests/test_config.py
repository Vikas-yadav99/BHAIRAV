"""Regression tests for config parsing (audit fix).

Verifies that AppConfig.from_dict() correctly parses ALL config sections
from Phases 1-27, not just the original Phase 1-11 fields.
"""
from bhairav.config import (
    AppConfig, AnalyticsConfig, EdgeConfig, FederationConfig,
    TrafficConfig, InvestigationConfig, NLPConfig, HAConfig,
    ComplianceConfig, ResponseConfig, load_config,
)


class TestAppConfigParsing:
    """AppConfig.from_dict must parse every phase's config from YAML overrides."""

    def test_analytics_config_parsing(self):
        cfg = AppConfig.from_dict({
            "analytics": {
                "enabled": False,
                "officer_pool": 20,
                "hotspot_min_alerts": 5,
                "forecast_horizon_sec": 30.0,
            }
        })
        assert cfg.analytics.enabled is False
        assert cfg.analytics.officer_pool == 20
        assert cfg.analytics.hotspot_min_alerts == 5
        assert cfg.analytics.forecast_horizon_sec == 30.0

    def test_edge_config_parsing(self):
        cfg = AppConfig.from_dict({
            "edge": {
                "enabled": True,
                "upstream_url": "https://edge.example.com",
                "mqtt_broker": "192.168.1.100",
                "mqtt_port": 1884,
                "fps_cap": 15,
            }
        })
        assert cfg.edge.enabled is True
        assert cfg.edge.upstream_url == "https://edge.example.com"
        assert cfg.edge.mqtt_broker == "192.168.1.100"
        assert cfg.edge.mqtt_port == 1884
        assert cfg.edge.fps_cap == 15

    def test_federation_config_parsing(self):
        cfg = AppConfig.from_dict({
            "federation": {
                "enabled": True,
                "site_id": "city-west",
                "peers": ["https://peer1:8000", "https://peer2:8000"],
                "secret": "shared-secret-123",
                "push_interval_sec": 5.0,
            }
        })
        assert cfg.federation.enabled is True
        assert cfg.federation.site_id == "city-west"
        assert len(cfg.federation.peers) == 2
        assert cfg.federation.secret == "shared-secret-123"
        assert cfg.federation.push_interval_sec == 5.0

    def test_ha_config_parsing(self):
        cfg = AppConfig.from_dict({
            "ha": {
                "enabled": True,
                "redis_url": "redis://cluster:6379",
                "heartbeat_interval": 3.0,
                "failure_threshold": 5,
                "balancer_strategy": "round_robin",
            }
        })
        assert cfg.ha.enabled is True
        assert cfg.ha.redis_url == "redis://cluster:6379"
        assert cfg.ha.heartbeat_interval == 3.0
        assert cfg.ha.failure_threshold == 5
        assert cfg.ha.balancer_strategy == "round_robin"

    def test_compliance_config_parsing(self):
        cfg = AppConfig.from_dict({
            "compliance": {
                "enabled": False,
                "evidence_retention_days": 60,
                "alert_retention_days": 180,
                "consent_store_path": "/data/consent.json",
            }
        })
        assert cfg.compliance.enabled is False
        assert cfg.compliance.evidence_retention_days == 60
        assert cfg.compliance.alert_retention_days == 180
        assert cfg.compliance.consent_store_path == "/data/consent.json"

    def test_traffic_config_parsing(self):
        cfg = AppConfig.from_dict({
            "traffic": {
                "enabled": False,
                "speed_threshold_free": 55.0,
                "window_sec": 600.0,
            }
        })
        assert cfg.traffic.enabled is False
        assert cfg.traffic.speed_threshold_free == 55.0
        assert cfg.traffic.window_sec == 600.0

    def test_investigation_config_parsing(self):
        cfg = AppConfig.from_dict({
            "investigation": {
                "max_events": 5000,
                "store_path": "/data/cases.json",
            }
        })
        assert cfg.investigation.max_events == 5000
        assert cfg.investigation.store_path == "/data/cases.json"

    def test_nlp_config_parsing(self):
        cfg = AppConfig.from_dict({"nlp": {"enabled": False}})
        assert cfg.nlp.enabled is False

    def test_response_config_parsing(self):
        cfg = AppConfig.from_dict({
            "response": {
                "ptz": {"enabled": True, "protocol": "onvif"},
                "reports": {"output_dir": "/data/reports"},
            }
        })
        assert cfg.response.ptz_enabled is True
        assert cfg.response.ptz_protocol == "onvif"
        assert cfg.response.reports_dir == "/data/reports"

    def test_all_configs_independently(self):
        """All 9 new config sections parse simultaneously."""
        cfg = AppConfig.from_dict({
            "analytics": {"enabled": False, "officer_pool": 5},
            "edge": {"enabled": True, "upstream_url": "https://x"},
            "federation": {"enabled": True, "site_id": "test"},
            "ha": {"enabled": True, "redis_url": "redis://x"},
            "compliance": {"enabled": False},
            "traffic": {"enabled": False},
            "investigation": {"max_events": 100},
            "nlp": {"enabled": False},
            "response": {"ptz": {"enabled": True}},
        })
        assert cfg.analytics.officer_pool == 5
        assert cfg.edge.upstream_url == "https://x"
        assert cfg.federation.site_id == "test"
        assert cfg.ha.redis_url == "redis://x"
        assert cfg.compliance.enabled is False
        assert cfg.traffic.enabled is False
        assert cfg.investigation.max_events == 100
        assert cfg.nlp.enabled is False
        assert cfg.response.ptz_enabled is True

    def test_defaults_when_empty(self):
        """Empty dict gives all defaults."""
        cfg = AppConfig.from_dict({})
        assert cfg.analytics.enabled is True
        assert cfg.edge.enabled is False
        assert cfg.federation.enabled is False
        assert cfg.ha.enabled is False
        assert cfg.compliance.enabled is True
        assert cfg.traffic.enabled is True
        assert cfg.investigation.enabled is True
        assert cfg.nlp.enabled is True

    def test_load_config_merges_defaults(self):
        """load_config returns valid config even without YAML file."""
        cfg = load_config("nonexistent.yaml")
        assert cfg.detector == "blob"
        assert cfg.analytics.enabled is True
        assert cfg.ha.enabled is False
