"""Tests for Phases 5, 6, 7, 8 — Phone Gateway, Analytics, Dashboard, Deployment."""
import sys, os, time, csv, io, json, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from bhairav.incidents import IncidentStore, DispatchEngine
from bhairav.phone_gateway import (
    SMSGateway, WhatsAppGateway, IVRSystem, PhoneVerifier,
    PhoneGateway, parse_sms_message,
)
from bhairav.city_analytics import (
    AnalyticsEngine, TrendAnalyzer, HeatmapGenerator,
    OfficerAnalytics, ReportExporter,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store():
    d = tempfile.mkdtemp()
    store = IncidentStore(path=os.path.join(d, "incidents"))
    yield store
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def engine(tmp_store):
    return DispatchEngine(tmp_store)


@pytest.fixture
def seed_data(tmp_store, engine):
    """Seed officers and incidents for analytics tests."""
    officers = []
    for i, (name, role, phone, lat, lng) in enumerate([
        ("Raj", "police", "+91-111", 28.6139, 77.2090),
        ("Priya", "medical", "+91-222", 28.6150, 77.2100),
        ("Amit", "fire", "+91-333", 28.6120, 77.2080),
    ]):
        off = tmp_store.register_officer(name, role, phone, lat, lng, [role])
        officers.append(off)

    # Create incidents at different times and categories
    incidents = []
    now = time.time()
    for i, (cat, level, lat, lng, hours_ago) in enumerate([
        ("crime", 3, 28.6139, 77.2090, 0.5),
        ("medical", 4, 28.6150, 77.2100, 2),
        ("fire", 2, 28.6120, 77.2080, 5),
        ("crime", 1, 28.6160, 77.2110, 25),
        ("medical", 3, 28.6110, 77.2070, 50),
        ("road_accident", 4, 28.6140, 77.2095, 0.1),
    ]):
        inc = tmp_store.create_incident(
            cat, level, lat, lng, f"Location {i}", f"Test incident {i}",
            source="public" if i % 2 == 0 else "camera",
        )
        # Backdate
        inc.created_at = now - hours_ago * 3600
        tmp_store._save_incidents()
        # Dispatch some
        if level >= 2:
            engine.dispatch(inc)
        incidents.append(inc)

    return {"officers": officers, "incidents": incidents}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: SMS PARSING
# ══════════════════════════════════════════════════════════════════════════════

class TestSMSMessageParser:

    def test_crime_detection(self):
        result = parse_sms_message("fight near bus stop urgent")
        assert result["category"] == "crime"
        assert result["emergency_level"] >= 3

    def test_medical_detection(self):
        result = parse_sms_message("heart attack person collapsed")
        assert result["category"] == "medical"
        assert result["emergency_level"] >= 1

    def test_fire_detection(self):
        result = parse_sms_message("fire in building smoke everywhere")
        assert result["category"] == "fire"
        assert result["emergency_level"] >= 3

    def test_accident_detection(self):
        result = parse_sms_message("car crash on main road")
        assert result["category"] == "road_accident"

    def test_default_severity(self):
        result = parse_sms_message("something happened")
        assert result["category"] == "other"
        assert result["emergency_level"] == 2

    def test_empty_message(self):
        result = parse_sms_message("")
        assert result["category"] == "other"
        assert result["emergency_level"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: SMS GATEWAY
# ══════════════════════════════════════════════════════════════════════════════

class TestSMSGateway:

    def test_send_sms(self):
        gw = SMSGateway()
        result = gw.send("+91-9876543210", "Emergency alert", priority="high")
        assert result["status"] == "sent"
        assert result["to"] == "+91-9876543210"
        assert gw.stats()["sent"] == 1

    def test_receive_sms(self):
        gw = SMSGateway()
        result = gw.receive("+91-9876543210", "fire urgent at market")
        assert result["parsed"]["category"] == "fire"
        assert gw.stats()["received"] == 1

    def test_recent_history(self):
        gw = SMSGateway()
        gw.send("+91-111", "msg1")
        gw.send("+91-222", "msg2")
        recent = gw.get_recent_sent(limit=1)
        assert len(recent) == 1
        assert recent[0]["to"] == "+91-222"


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: WHATSAPP GATEWAY
# ══════════════════════════════════════════════════════════════════════════════

class TestWhatsAppGateway:

    def test_send_text(self):
        wa = WhatsAppGateway()
        result = wa.send_text("+91-9876543210", "Hello from BHAIRAV")
        assert result["type"] == "text"
        assert wa.stats()["sent"] == 1

    def test_send_location(self):
        wa = WhatsAppGateway()
        result = wa.send_location("+91-9876543210", 28.6139, 77.2090,
                                   name="Incident Location")
        assert result["type"] == "location"
        assert result["location"]["lat"] == 28.6139

    def test_send_alert(self):
        wa = WhatsAppGateway()
        incident = {
            "category": "crime", "emergency_level": 4,
            "location_name": "Main St", "description": "Fight",
        }
        result = wa.send_alert("+91-9876543210", incident)
        assert result["type"] == "text"
        assert "CRITICAL" in result["text"] or "Level 4" in result["text"]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: OTP VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

class TestPhoneVerification:

    def test_generate_otp(self):
        pv = PhoneVerifier()
        result = pv.generate_otp("+91-9876543210")
        assert result["expires_in"] == 300
        assert "+91-9876543210" in pv._pending

    def test_verify_correct_otp(self):
        pv = PhoneVerifier()
        pv.generate_otp("+91-9876543210")
        otp = pv._pending["+91-9876543210"]["otp"]
        result = pv.verify_otp("+91-9876543210", otp)
        assert result["verified"] is True

    def test_verify_wrong_otp(self):
        pv = PhoneVerifier()
        pv.generate_otp("+91-9876543210")
        result = pv.verify_otp("+91-9876543210", "000000")
        assert result["verified"] is False
        assert "attempts_remaining" in result

    def test_verify_no_pending(self):
        pv = PhoneVerifier()
        result = pv.verify_otp("+91-9999999999", "123456")
        assert result["verified"] is False
        assert "No OTP" in result["error"]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: IVR SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class TestIVRSystem:

    def test_start_call(self):
        ivr = IVRSystem()
        result = ivr.start_call("+91-9876543210")
        assert "call_id" in result
        assert "greeting" in result
        assert ivr.stats()["calls_received"] == 1

    def test_category_selection(self):
        ivr = IVRSystem()
        call = ivr.start_call("+91-9876543210")
        result = ivr.process_input(call["call_id"], "3")  # Crime
        assert "severity" in result["prompt"].lower() or "select" in result["prompt"].lower()

    def test_full_ivr_flow(self):
        ivr = IVRSystem()
        call = ivr.start_call("+91-9876543210")
        cid = call["call_id"]

        # Category: 3 = crime
        r1 = ivr.process_input(cid, "3")
        assert "severity" in r1["prompt"].lower() or "press 1" in r1["prompt"].lower()

        # Severity: 4 = critical
        r2 = ivr.process_input(cid, "4")
        assert "describe" in r2["prompt"].lower()

        # Description
        r3 = ivr.process_input(cid, "fight")
        r4 = ivr.process_input(cid, "near")
        r5 = ivr.process_input(cid, "station")
        r6 = ivr.process_input(cid, "#")
        assert "confirm" in r6["prompt"].lower()

        # Confirm
        r7 = ivr.process_input(cid, "1")
        assert r7["completed"] is True
        assert r7["incident"]["category"] == "crime"
        assert r7["incident"]["emergency_level"] == 4
        assert ivr.stats()["incidents_created"] == 1

    def test_cancel_ivr(self):
        ivr = IVRSystem()
        call = ivr.start_call("+91-9876543210")
        cid = call["call_id"]
        ivr.process_input(cid, "1")  # medical
        ivr.process_input(cid, "2")  # medium
        ivr.process_input(cid, "#")  # done describing
        r = ivr.process_input(cid, "2")  # cancel
        assert "cancel" in r["message"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: UNIFIED PHONE GATEWAY
# ══════════════════════════════════════════════════════════════════════════════

class TestPhoneGateway:

    def test_receive_sms_report(self):
        pg = PhoneGateway()
        result = pg.receive_sms("+91-9876543210", "fire urgent at temple")
        assert result["parsed"]["category"] == "fire"
        assert result["channel"] == "sms"
        assert pg.stats()["total_reports"] == 1

    def test_receive_whatsapp_report(self):
        pg = PhoneGateway()
        result = pg.receive_whatsapp("+91-9876543210", "accident on highway")
        assert result["parsed"]["category"] == "road_accident"
        assert result["channel"] == "whatsapp"

    def test_start_ivr(self):
        pg = PhoneGateway()
        result = pg.start_ivr_call("+91-9876543210")
        assert "call_id" in result

    def test_otp_flow(self):
        pg = PhoneGateway()
        pg.send_otp("+91-9876543210")
        otp = pg.verifier._pending["+91-9876543210"]["otp"]
        result = pg.verify_phone("+91-9876543210", otp)
        assert result["verified"] is True


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

class TestTrendAnalysis:

    def test_hourly_trend(self, seed_data):
        store = seed_data["officers"]  # reuse fixture
        from bhairav.incidents import IncidentStore
        # seed_data fixture already created incidents
        analyzer = TrendAnalyzer()
        # Get incidents as dicts
        inc_dicts = [i.to_dict() for i in seed_data["incidents"]]
        trend = analyzer.hourly_trend(inc_dicts, hours=48)
        assert len(trend) == 48
        assert all("time" in t for t in trend)
        assert all("count" in t for t in trend)

    def test_peak_hours(self, seed_data):
        analyzer = TrendAnalyzer()
        inc_dicts = [i.to_dict() for i in seed_data["incidents"]]
        peak = analyzer.peak_hours(inc_dicts)
        assert len(peak) > 0
        assert peak[0]["count"] >= peak[-1]["count"]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

class TestHeatmap:

    def test_generate_heatmap(self, seed_data):
        gen = HeatmapGenerator(cell_size_deg=0.01)
        inc_dicts = [i.to_dict() for i in seed_data["incidents"]]
        heatmap = gen.generate(inc_dicts)
        assert len(heatmap) > 0
        assert all("count" in h for h in heatmap)
        assert all("lat" in h for h in heatmap)

    def test_empty_heatmap(self):
        gen = HeatmapGenerator()
        heatmap = gen.generate([])
        assert heatmap == []


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: OFFICER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

class TestOfficerAnalytics:

    def test_officer_stats(self, tmp_store, seed_data):
        oa = OfficerAnalytics(tmp_store)
        stats = oa.get_officer_stats(hours=168)
        assert len(stats) == 3
        assert all("name" in s for s in stats)
        assert all("resolution_rate" in s for s in stats)

    def test_team_summary(self, tmp_store, seed_data):
        oa = OfficerAnalytics(tmp_store)
        summary = oa.get_team_summary(hours=168)
        assert summary["total_officers"] == 3
        assert "avg_resolution_rate" in summary


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: REPORT EXPORT
# ══════════════════════════════════════════════════════════════════════════════

class TestReportExport:

    def test_csv_export(self, seed_data):
        exporter = ReportExporter()
        inc_dicts = [i.to_dict() for i in seed_data["incidents"]]
        csv_str = exporter.to_csv(inc_dicts)
        assert csv_str
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 6
        assert "id" in rows[0]
        assert "category" in rows[0]

    def test_json_export(self, seed_data):
        exporter = ReportExporter()
        inc_dicts = [i.to_dict() for i in seed_data["incidents"]]
        report = exporter.to_json_report(inc_dicts, "Test Report")
        assert report["report_title"] == "Test Report"
        assert report["total_incidents"] == 6
        assert "by_category" in report["summary"]

    def test_geojson_export(self, seed_data):
        exporter = ReportExporter()
        inc_dicts = [i.to_dict() for i in seed_data["incidents"]]
        geo = exporter.to_geojson(inc_dicts)
        assert geo["type"] == "FeatureCollection"
        assert len(geo["features"]) == 6
        assert geo["features"][0]["geometry"]["type"] == "Point"

    def test_empty_export(self):
        exporter = ReportExporter()
        assert exporter.to_csv([]) == ""
        report = exporter.to_json_report([])
        assert report["total_incidents"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: UNIFIED ANALYTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyticsEngine:

    def test_full_analytics(self, tmp_store, seed_data):
        engine = AnalyticsEngine(tmp_store)
        analytics = engine.get_full_analytics(hours=168)
        assert "total_incidents" in analytics
        assert "hourly_trend" in analytics
        assert "heatmap" in analytics
        assert "officer_performance" in analytics
        assert "team_summary" in analytics
        assert "peak_hours" in analytics

    def test_export_csv(self, tmp_store, seed_data):
        engine = AnalyticsEngine(tmp_store)
        csv_str = engine.export_csv(hours=168)
        assert csv_str
        assert "crime" in csv_str or "medical" in csv_str

    def test_export_geojson(self, tmp_store, seed_data):
        engine = AnalyticsEngine(tmp_store)
        geo = engine.export_geojson(hours=168)
        assert geo["type"] == "FeatureCollection"

    def test_alert_patterns(self, tmp_store, seed_data):
        engine = AnalyticsEngine(tmp_store)
        patterns = engine.get_alert_patterns()
        assert "peak_hour" in patterns
        assert "hotspot_hours" in patterns
