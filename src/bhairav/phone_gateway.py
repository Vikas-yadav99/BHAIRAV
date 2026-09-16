"""BHAIRAV Phone Gateway — SMS, WhatsApp, and IVR reporting.

Phase 5: Allows anyone to report incidents via:
- SMS to a dedicated number (Twilio / MSG91 / TextBelt)
- WhatsApp Business API
- IVR (Interactive Voice Response) for feature phones
- Phone number verification via OTP

This fills the gap for people without smartphones or internet access.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from collections import Counter
from dataclasses import dataclass
log = logging.getLogger("bhairav.phone_gateway")


# ── Category/Level Parsing from SMS ────────────────────────────────────────

SMS_KEYWORDS = {
    # Category keywords
    "medical": ["medical", "heart", "attack", "injury", "ambulance", "doctor", "sick", "blood", "unconscious", "collapse"],
    "fire": ["fire", "burn", "smoke", "flame", "blast", "explosion", "lpg", "gas"],
    "crime": ["crime", "theft", "robbery", "fight", "assault", "murder", "kidnap", "rape", "attack", "gun", "weapon", "stab"],
    "road_accident": ["accident", "crash", "collision", "hit", "run", "vehicle", "car", "bike", "truck", "road"],
    "disaster": ["flood", "earthquake", "landslide", "storm", "disaster", "collapse", "building"],
    "missing_person": ["missing", "lost", "kidnap", "abduct", "child"],
}

SEVERITY_KEYWORDS = {
    4: ["urgent", "critical", "emergency", "dying", "bleeding", "fire", "shooting", "gun", "rape", "kidnap"],
    3: ["serious", "injured", "fight", "robbery", "accident", "smoke"],
    2: ["minor", "small", "low", "theft", "suspicious"],
    1: ["info", "aware", "notice", "noise"],
}


def parse_sms_message(text: str) -> dict:
    """Parse an SMS message to extract category, severity, and location hints.

    Returns:
        {
            "category": "crime",
            "emergency_level": 3,
            "keywords_found": ["fight", "urgent"],
            "raw_text": "fight urgent near market",
        }
    """
    text_lower = text.lower().strip()

    # Detect category
    scores = {}
    for cat, keywords in SMS_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[cat] = score

    category = max(scores, key=scores.get) if scores else "other"

    # Detect severity
    sev_scores = {}
    for level, keywords in SEVERITY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            sev_scores[level] = score

    emergency_level = max(sev_scores, key=sev_scores.get) if sev_scores else 2

    # Find matched keywords
    all_keywords = []
    for keywords in list(SMS_KEYWORDS.values()) + list(SEVERITY_KEYWORDS.values()):
        all_keywords.extend([kw for kw in keywords if kw in text_lower])

    return {
        "category": category,
        "emergency_level": emergency_level,
        "keywords_found": list(set(all_keywords)),
        "raw_text": text,
    }


# ── OTP Verification ────────────────────────────────────────────────────────

class PhoneVerifier:
    """Phone number verification via OTP.

    In production: sends OTP via SMS gateway, verifies on callback.
    In demo: OTP is always 123456 (or generated and logged).
    """

    OTP_LENGTH = 6
    OTP_TTL_SEC = 300  # 5 minutes
    MAX_ATTEMPTS = 3

    def __init__(self, sms_gateway=None):
        self.sms_gateway = sms_gateway
        self._pending: dict[str, dict] = {}  # phone -> {otp, expires, attempts}

    def generate_otp(self, phone: str) -> dict:
        """Generate and send OTP to phone number."""
        import random
        otp = "".join([str(random.randint(0, 9)) for _ in range(self.OTP_LENGTH)])
        expires = time.time() + self.OTP_TTL_SEC

        self._pending[phone] = {
            "otp": otp,
            "expires": expires,
            "attempts": 0,
            "created_at": time.time(),
        }

        # Send via gateway
        if self.sms_gateway:
            self.sms_gateway.send(phone, f"Your BHAIRAV verification code: {otp}")
        else:
            log.info("OTP sent to %s (gateway stub, OTP not logged for security)", phone)

        return {
            "phone": phone,
            "expires_in": self.OTP_TTL_SEC,
            "message": "OTP sent successfully",
        }

    def verify_otp(self, phone: str, otp: str) -> dict:
        """Verify an OTP."""
        pending = self._pending.get(phone)
        if not pending:
            return {"verified": False, "error": "No OTP requested for this number"}

        if time.time() > pending["expires"]:
            del self._pending[phone]
            return {"verified": False, "error": "OTP expired"}

        pending["attempts"] += 1
        if pending["attempts"] > self.MAX_ATTEMPTS:
            del self._pending[phone]
            return {"verified": False, "error": "Too many attempts"}

        if otp == pending["otp"]:
            del self._pending[phone]
            return {"verified": True, "phone": phone}

        return {"verified": False, "error": "Invalid OTP",
                "attempts_remaining": self.MAX_ATTEMPTS - pending["attempts"]}

    def is_verified_recently(self, phone: str, within_sec: float = 3600) -> bool:
        """Check if phone was verified recently."""
        # In production: check verification database
        # For now: trust that if they got past OTP, they're verified
        return phone not in self._pending  # pending means unverified


# ── SMS Gateway Stub ────────────────────────────────────────────────────────


class SMSGateway:
    """Real SMS gateway with Twilio / MSG91 / TextBelt support.

    Auto-detects provider from constructor args or environment variables:
        TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN + TWILIO_FROM_NUMBER
        MSG91_API_KEY + MSG91_SENDER_ID
        TEXTBELT_API_KEY

    Falls back to stub (log-only) when no credentials are configured.

    Handles:
    - Outbound SMS (alerts to officers, OTP to reporters)
    - Inbound SMS (incident reports from public)
    - Delivery status tracking with retry
    """

    def __init__(self, provider: str | None = None, api_key: str | None = None,
                 from_number: str | None = None,
                 account_sid: str | None = None, auth_token: str | None = None):
        self.provider = provider or self._detect_provider()
        self.api_key = api_key or os.environ.get("TEXTBELT_API_KEY") or os.environ.get("MSG91_API_KEY")
        self.account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.environ.get("TWILIO_FROM_NUMBER", "+91-BHAIRAV")
        self.msg91_sender = os.environ.get("MSG91_SENDER_ID", "BHAIRAV")
        self._sent: list[dict] = []
        self._received: list[dict] = []
        self._stats = {"sent": 0, "received": 0, "failed": 0}

    @staticmethod
    def _detect_provider() -> str:
        if os.environ.get("TWILIO_ACCOUNT_SID"):
            return "twilio"
        if os.environ.get("MSG91_API_KEY"):
            return "msg91"
        if os.environ.get("TEXTBELT_API_KEY"):
            return "textbelt"
        return "stub"

    def send(self, to: str, message: str, priority: str = "normal") -> dict:
        """Send an SMS via the configured provider with retry."""
        now = time.time()
        sms = {
            "id": uuid.uuid4().hex[:12],
            "to": to,
            "from": self.from_number,
            "message": message,
            "priority": priority,
            "sent_at": now,
            "status": "pending",
            "provider": self.provider,
        }

        for attempt in range(3):
            try:
                if self.provider == "twilio":
                    result = self._send_twilio(to, message)
                elif self.provider == "msg91":
                    result = self._send_msg91(to, message)
                elif self.provider == "textbelt":
                    result = self._send_textbelt(to, message)
                else:
                    log.info("SMS [%s] (stub) -> %s: %s", priority, to, message[:60])
                    result = {"ok": True}

                if result.get("ok"):
                    sms["status"] = "sent"
                    self._stats["sent"] += 1
                    break
                else:
                    sms["status"] = "failed"
                    sms["error"] = result.get("error", "unknown")
                    log.warning("SMS attempt %d failed: %s", attempt + 1, result.get("error"))
            except Exception as exc:
                sms["status"] = "failed"
                sms["error"] = str(exc)
                log.warning("SMS attempt %d exception: %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
        else:
            self._stats["failed"] += 1
            sms["status"] = "failed"

        self._sent.append(sms)
        return sms

    def _send_twilio(self, to: str, message: str) -> dict:
        """Send SMS via Twilio REST API."""
        import httpx
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        resp = httpx.post(
            url,
            auth=(self.account_sid, self.auth_token),
            data={"To": to, "From": self.from_number, "Body": message},
            timeout=15.0,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            log.info("Twilio SMS sent: %s -> %s (sid=%s)", self.from_number, to, data.get("sid"))
            return {"ok": True, "sid": data.get("sid")}
        log.error("Twilio error %d: %s", resp.status_code, resp.text[:200])
        return {"ok": False, "error": f"Twilio {resp.status_code}: {resp.text[:200]}"}

    def _send_msg91(self, to: str, message: str) -> dict:
        """Send SMS via MSG91 API (India-focused)."""
        import httpx
        phone = to.lstrip("+")
        if phone.startswith("91") and len(phone) == 12:
            phone = phone[2:]
        url = "https://api.msg91.com/api/v5/flow"
        payload = {
            "flow_id": os.environ.get("MSG91_FLOW_ID", ""),
            "mobiles": f"91{phone}",
            "VAR1": message[:160],
        }
        headers = {"authkey": self.api_key or "", "Content-Type": "application/json"}
        resp = httpx.post(url, json=payload, headers=headers, timeout=15.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("type") == "success":
                log.info("MSG91 SMS sent to %s", phone)
                return {"ok": True, "request_id": data.get("request_id")}
        log.error("MSG91 error %d: %s", resp.status_code, resp.text[:200])
        return {"ok": False, "error": f"MSG91 {resp.status_code}: {resp.text[:200]}"}

    def _send_textbelt(self, to: str, message: str) -> dict:
        """Send SMS via TextBelt API (limited free tier)."""
        import httpx
        resp = httpx.post(
            "https://textbelt.com/text",
            data={"phone": to, "message": message, "key": self.api_key or "textbelt"},
            timeout=15.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                log.info("TextBelt SMS sent to %s (quota: %s)", to, data.get("quotaRemaining"))
                return {"ok": True, "quota": data.get("quotaRemaining")}
        return {"ok": False, "error": f"TextBelt {resp.status_code}"}

    @property
    def is_real(self) -> bool:
        """True if a real provider is configured (not stub)."""
        return self.provider != "stub"

    def receive(self, from_number: str, body: str) -> dict:
        """Process an incoming SMS (incident report)."""
        now = time.time()
        sms = {
            "id": uuid.uuid4().hex[:12],
            "from": from_number,
            "body": body,
            "received_at": now,
        }
        self._received.append(sms)
        self._stats["received"] += 1

        parsed = parse_sms_message(body)
        sms["parsed"] = parsed

        log.info("SMS received from %s: category=%s level=%d",
                 from_number, parsed["category"], parsed["emergency_level"])

        return sms

    def get_recent_sent(self, limit: int = 50) -> list[dict]:
        return self._sent[-limit:]

    def get_recent_received(self, limit: int = 50) -> list[dict]:
        return self._received[-limit:]

    def stats(self) -> dict:
        return dict(self._stats)
# ── WhatsApp Gateway Stub ──────────────────────────────────────────────────

class WhatsAppGateway:
    """WhatsApp Business API stub.

    In production: uses Meta's WhatsApp Business API.
    Supports: text messages, location sharing, quick replies.
    """

    def __init__(self, api_key: str | None = None, phone_number_id: str = ""):
        self.api_key = api_key
        self.phone_number_id = phone_number_id
        self._sent: list[dict] = []
        self._stats = {"sent": 0, "received": 0, "failed": 0}

    def send_text(self, to: str, text: str) -> dict:
        """Send a text message via WhatsApp."""
        msg = {
            "id": uuid.uuid4().hex[:12],
            "to": to,
            "type": "text",
            "text": text,
            "sent_at": time.time(),
        }
        self._sent.append(msg)
        self._stats["sent"] += 1
        log.info("WhatsApp → %s: %s", to, text[:60])
        return msg

    def send_location(self, to: str, lat: float, lng: float,
                      name: str = "", address: str = "") -> dict:
        """Send a location message via WhatsApp."""
        msg = {
            "id": uuid.uuid4().hex[:12],
            "to": to,
            "type": "location",
            "location": {"lat": lat, "lng": lng, "name": name, "address": address},
            "sent_at": time.time(),
        }
        self._sent.append(msg)
        self._stats["sent"] += 1
        return msg

    def send_alert(self, to: str, incident: dict) -> dict:
        """Send a formatted incident alert via WhatsApp."""
        cat = incident.get("category", "unknown")
        level = incident.get("emergency_level", 1)
        loc = incident.get("location_name", "Unknown location")
        desc = incident.get("description", "")

        emoji = {1: "ℹ️", 2: "⚠️", 3: "🔶", 4: "🔴"}.get(level, "📋")
        text = (
            f"{emoji} BHAIRAV Alert — Level {level}\n\n"
            f"Category: {cat.upper()}\n"
            f"Location: {loc}\n"
            f"Details: {desc}\n\n"
            f"Please respond immediately."
        )
        return self.send_text(to, text)

    def stats(self) -> dict:
        return dict(self._stats)


# ── IVR System ──────────────────────────────────────────────────────────────

class IVRSystem:
    """Interactive Voice Response for feature phones.

    Flow:
    1. User calls BHAIRAV number
    2. IVR: "Press 1 for Medical, 2 for Fire, 3 for Crime..."
    3. IVR: "Press 1 for Low, 2 for Medium, 3 for High, 4 for Critical"
    4. IVR: "Describe briefly. Press # when done."
    5. System creates incident from parsed input

    In production: integrates with Twilio Voice or similar.
    """

    MENUS = {
        "category": {
            "1": "medical",
            "2": "fire",
            "3": "crime",
            "4": "road_accident",
            "5": "disaster",
            "6": "missing_person",
            "7": "other",
        },
        "severity": {
            "1": 1,  # Low
            "2": 2,  # Medium
            "3": 3,  # High
            "4": 4,  # Critical
        },
    }

    SESSION_TTL_SEC = 300  # 5 minutes

    def __init__(self):
        self._calls: dict[str, dict] = {}  # call_id -> state
        self._stats = {"calls_received": 0, "incidents_created": 0}

    def start_call(self, caller_phone: str) -> dict:
        """Start a new IVR call session (cleans up expired sessions first)."""
        # Clean up expired sessions to prevent memory leak
        import time as _time
        expired = [cid for cid, s in self._calls.items()
                   if _time.time() - s.get("started_at", 0) > self.SESSION_TTL_SEC]
        for cid in expired:
            del self._calls[cid]
        if expired:
            log.info("Cleaned up %d expired IVR sessions", len(expired))

        call_id = uuid.uuid4().hex[:12]
        self._calls[call_id] = {
            "caller_phone": caller_phone,
            "started_at": time.time(),
            "step": "category",
            "category": None,
            "severity": None,
            "description": "",
            "status": "active",
        }
        self._stats["calls_received"] += 1

        return {
            "call_id": call_id,
            "greeting": "Welcome to BHAIRAV Emergency. "
                        "Press 1 for Medical, 2 for Fire, 3 for Crime, "
                        "4 for Road Accident, 5 for Disaster, "
                        "6 for Missing Person, 7 for Other.",
        }

    def process_input(self, call_id: str, input_key: str) -> dict:
        """Process a DTMF input from the caller."""
        call = self._calls.get(call_id)
        if not call or call["status"] != "active":
            return {"error": "Invalid or ended call"}
        # Check session expiry
        import time as _time
        if _time.time() - call.get("started_at", 0) > self.SESSION_TTL_SEC:
            del self._calls[call_id]
            return {"error": "Call session expired. Please call again."}

        step = call["step"]

        if step == "category":
            category = self.MENUS["category"].get(input_key)
            if not category:
                return {"prompt": "Invalid option. Press 1-7."}
            call["category"] = category
            call["step"] = "severity"
            return {
                "prompt": f"Category: {category}. "
                          "Now select severity: "
                          "1 for Low, 2 for Medium, 3 for High, 4 for Critical.",
            }

        elif step == "severity":
            severity = self.MENUS["severity"].get(input_key)
            if not severity:
                return {"prompt": "Invalid option. Press 1-4."}
            call["severity"] = severity
            call["step"] = "description"
            return {
                "prompt": "Please describe the incident briefly. "
                          "Press # when done.",
            }

        elif step == "description":
            if input_key == "#":
                call["step"] = "confirm"
                return {
                    "prompt": f"Confirm: {call['category']} incident, "
                              f"severity level {call['severity']}. "
                              "Press 1 to confirm, 2 to cancel.",
                }
            else:
                call["description"] += input_key + " "
                return {"prompt": "Received. Press # when done."}

        elif step == "confirm":
            if input_key == "1":
                call["status"] = "completed"
                self._stats["incidents_created"] += 1
                return {
                    "completed": True,
                    "incident": {
                        "category": call["category"],
                        "emergency_level": call["severity"],
                        "description": call["description"].strip(),
                        "caller_phone": call["caller_phone"],
                        "source": "ivr",
                    },
                    "message": "Incident reported. Help is on the way. Stay safe.",
                }
            elif input_key == "2":
                call["status"] = "cancelled"
                return {"completed": True, "message": "Cancelled. Stay safe."}

        return {"prompt": "Invalid input."}

    def get_call(self, call_id: str) -> dict | None:
        return self._calls.get(call_id)

    def stats(self) -> dict:
        return dict(self._stats)


# ── Unified Phone Gateway ───────────────────────────────────────────────────

class PhoneGateway:
    """Unified phone gateway combining SMS, WhatsApp, and IVR.

    Usage:
        gateway = PhoneGateway()
        # SMS report
        result = gateway.receive_sms("+91-9876543210", "fire urgent near market")
        # IVR report
        call = gateway.start_ivr_call("+91-9876543211")
        # Verify phone
        gateway.send_otp("+91-9876543212")
    """

    def __init__(self, sms_gateway: SMSGateway | None = None,
                 whatsapp: WhatsAppGateway | None = None):
        self.sms = sms_gateway or SMSGateway()
        self.whatsapp = whatsapp or WhatsAppGateway()
        self.ivr = IVRSystem()
        self.verifier = PhoneVerifier(self.sms)
        self._reports: list[dict] = []
        self._stats = {"total_reports": 0, "by_channel": Counter()}

    def receive_sms(self, phone: str, message: str) -> dict:
        """Process incoming SMS report."""
        parsed = self.sms.receive(phone, message)
        report = {
            "id": uuid.uuid4().hex[:12],
            "phone": phone,
            "channel": "sms",
            "message": message,
            "parsed": parsed["parsed"],
            "received_at": parsed["received_at"],
        }
        self._reports.append(report)
        self._stats["total_reports"] += 1
        self._stats["by_channel"]["sms"] += 1
        return report

    def receive_whatsapp(self, phone: str, message: str) -> dict:
        """Process incoming WhatsApp report."""
        parsed = parse_sms_message(message)
        report = {
            "id": uuid.uuid4().hex[:12],
            "phone": phone,
            "channel": "whatsapp",
            "message": message,
            "parsed": parsed,
            "received_at": time.time(),
        }
        self._reports.append(report)
        self._stats["total_reports"] += 1
        self._stats["by_channel"]["whatsapp"] += 1
        return report

    def start_ivr_call(self, phone: str) -> dict:
        """Start an IVR call session."""
        return self.ivr.start_call(phone)

    def send_otp(self, phone: str) -> dict:
        """Send OTP for phone verification."""
        return self.verifier.generate_otp(phone)

    def verify_phone(self, phone: str, otp: str) -> dict:
        """Verify phone with OTP."""
        return self.verifier.verify_otp(phone, otp)

    def get_reports(self, limit: int = 50) -> list[dict]:
        return self._reports[-limit:]

    def stats(self) -> dict:
        return {
            "total_reports": self._stats["total_reports"],
            "by_channel": dict(self._stats["by_channel"]),
            "sms": self.sms.stats(),
            "whatsapp": self.whatsapp.stats(),
            "ivr": self.ivr.stats(),
        }
