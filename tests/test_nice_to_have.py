"""Tests for nice-to-have items: multi-city, NLP chatbot, calibration, wireframes."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestMultiCity:
    """Test multi-city support."""

    def test_list_available_cities(self):
        from bhairav.city_loader import list_available_cities
        cities = list_available_cities()
        names = [c["name"] for c in cities]
        assert "Indore" in names
        assert len(cities) >= 2

    def test_load_bhopal_config(self):
        from bhairav.city_loader import get_city_config
        cfg = get_city_config("bhopal")
        assert cfg is not None
        assert cfg["city"] == "Bhopal"
        assert cfg["state"] == "Madhya Pradesh"
        assert len(cfg["cameras"]) >= 5

    def test_load_jaipur_config(self):
        from bhairav.city_loader import get_city_config
        cfg = get_city_config("jaipur")
        assert cfg is not None
        assert cfg["city"] == "Jaipur"
        assert cfg["state"] == "Rajasthan"
        assert len(cfg["cameras"]) >= 5

    def test_city_has_zones(self):
        from bhairav.city_loader import get_city_config
        for name in ["indore", "bhopal", "jaipur"]:
            cfg = get_city_config(name)
            assert cfg is not None
            assert len(cfg["zones"]) >= 3

    def test_city_has_police_stations(self):
        from bhairav.city_loader import get_city_config
        cfg = get_city_config("indore")
        assert len(cfg.get("police_stations", [])) >= 5

    def test_nonexistent_city_returns_none(self):
        from bhairav.city_loader import get_city_config
        assert get_city_config("atlantis") is None


class TestNLPChatbot:
    """Test NLP query engine with location queries."""

    def test_parse_location_query(self):
        from bhairav.nlp import NLPQueryEngine
        engine = NLPQueryEngine()
        r = engine.query("show me crimes in sarafa bazaar today")
        assert "sarafa" in r.parsed.get("locations", [])

    def test_parse_time_query(self):
        from bhairav.nlp import NLPQueryEngine
        engine = NLPQueryEngine()
        r = engine.query("show me all fights today")
        assert r.parsed.get("time_range") is not None

    def test_parse_category_query(self):
        from bhairav.nlp import NLPQueryEngine
        engine = NLPQueryEngine()
        r = engine.query("how many fights this week")
        assert "fight" in r.parsed.get("rules", [])

    def test_parse_camera_query(self):
        from bhairav.nlp import NLPQueryEngine
        engine = NLPQueryEngine()
        r = engine.query("show me alerts on camera CAM-01")
        assert "CAM-01" in r.parsed.get("cameras", [])

    def test_parse_summary_intent(self):
        from bhairav.nlp import NLPQueryEngine
        engine = NLPQueryEngine()
        r = engine.query("give me a summary of today")
        assert r.parsed.get("intent") == "summary"

    def test_result_has_to_dict(self):
        from bhairav.nlp import NLPQueryEngine
        engine = NLPQueryEngine()
        r = engine.query("show me all crimes")
        d = r.to_dict()
        assert "query" in d
        assert "parsed" in d
        assert "count" in d


class TestCalibration:
    """Test camera calibration module."""

    def test_create_calibrator(self):
        from bhairav.calibration import CameraCalibrator
        cal = CameraCalibrator("TEST-CAM", 1920, 1080)
        assert cal.camera_id == "TEST-CAM"

    def test_add_calibration_points(self):
        from bhairav.calibration import CameraCalibrator
        cal = CameraCalibrator("TEST-CAM", 1920, 1080)
        cal.add_calibration_point(0, 0, 22.7200, 75.8500)
        cal.add_calibration_point(1920, 0, 22.7200, 75.8600)
        cal.add_calibration_point(1920, 1080, 22.7100, 75.8600)
        cal.add_calibration_point(0, 1080, 22.7100, 75.8500)
        assert len(cal._points) == 4

    def test_calibrate_with_4_points(self):
        from bhairav.calibration import CameraCalibrator
        cal = CameraCalibrator("TEST-CAM", 1920, 1080)
        cal.add_calibration_point(0, 0, 22.7200, 75.8500)
        cal.add_calibration_point(1920, 0, 22.7200, 75.8600)
        cal.add_calibration_point(1920, 1080, 22.7100, 75.8600)
        cal.add_calibration_point(0, 1080, 22.7100, 75.8500)
        result = cal.calibrate()
        assert result is True
        assert cal._calibrated is True

    def test_pixel_to_world(self):
        from bhairav.calibration import CameraCalibrator
        cal = CameraCalibrator("TEST-CAM", 1920, 1080)
        cal.add_calibration_point(0, 0, 22.7200, 75.8500)
        cal.add_calibration_point(1920, 0, 22.7200, 75.8600)
        cal.add_calibration_point(1920, 1080, 22.7100, 75.8600)
        cal.add_calibration_point(0, 1080, 22.7100, 75.8500)
        cal.calibrate()
        gps = cal.pixel_to_world(960, 540)
        assert gps is not None
        lat, lng = gps
        assert 22.710 <= lat <= 22.720
        assert 75.850 <= lng <= 75.860

    def test_to_dict_roundtrip(self):
        from bhairav.calibration import CameraCalibrator
        cal = CameraCalibrator("TEST-CAM", 1920, 1080)
        cal.add_calibration_point(0, 0, 22.7200, 75.8500, label="top-left")
        cal.add_calibration_point(1920, 0, 22.7200, 75.8600)
        cal.add_calibration_point(1920, 1080, 22.7100, 75.8600)
        cal.add_calibration_point(0, 1080, 22.7100, 75.8500)
        cal.calibrate()
        data = cal.to_dict()
        assert data["calibrated"] is True
        assert len(data["points"]) == 4
        # Restore from dict
        cal2 = CameraCalibrator.from_dict(data)
        assert cal2._calibrated is True
        gps = cal2.pixel_to_world(960, 540)
        assert gps is not None

    def test_insufficient_points(self):
        from bhairav.calibration import CameraCalibrator
        cal = CameraCalibrator("TEST-CAM", 1920, 1080)
        cal.add_calibration_point(0, 0, 22.72, 75.85)
        cal.add_calibration_point(1920, 0, 22.72, 75.86)
        assert cal.calibrate() is False

    def test_fov_polygon(self):
        from bhairav.calibration import CameraCalibrator
        cal = CameraCalibrator("TEST-CAM", 1920, 1080)
        cal.add_calibration_point(0, 0, 22.7200, 75.8500)
        cal.add_calibration_point(1920, 0, 22.7200, 75.8600)
        cal.add_calibration_point(1920, 1080, 22.7100, 75.8600)
        cal.add_calibration_point(0, 1080, 22.7100, 75.8500)
        cal.calibrate()
        poly = cal.get_fov_polygon()
        assert len(poly) == 4
        assert all("lat" in p and "lng" in p for p in poly)


class TestMobileWireframes:
    """Test that mobile wireframe HTML files exist and are valid."""

    def test_officer_wireframe_exists(self):
        path = Path("dashboard/mobile/officer.html")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "BHAIRAV" in content
        assert "Accept" in content
        assert "SOS" in content

    def test_report_wireframe_exists(self):
        path = Path("dashboard/mobile/report.html")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Report Incident" in content
        assert "Submit Report" in content

    def test_wireframes_have_viewport_meta(self):
        for name in ["officer.html", "report.html"]:
            content = Path(f"dashboard/mobile/{name}").read_text(encoding="utf-8")
            assert "viewport" in content
