"""Generate a self-signed TLS certificate for BHAIRAV HTTPS mode.

Usage:
    python scripts/make_cert.py                  # -> output/tls/cert.pem + key.pem
    python scripts/make_cert.py --host 192.168.1.10 --out-dir output/tls

Then run the server with:
    python scripts/serve.py --tls-cert output/tls/cert.pem --tls-key output/tls/key.pem

Uses the `cryptography` package (already a BHAIRAV dependency). For real
deployments replace this with a CA-issued certificate; this is for encrypting
traffic between the browser and the server so tokens/evidence never travel
in cleartext.
"""
from __future__ import annotations

import argparse
import ipaddress
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_cert(out_dir, hosts, days=365):
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SystemExit("cryptography is required: pip install cryptography") from exc

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "BHAIRAV TLS")])
    san_entries = []
    for h in hosts:
        try:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            san_entries.append(x509.DNSName(h))
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=days))
            .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
            .sign(key, hashes.SHA256()))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cert_path = out_dir / "cert.pem"
    key_path = out_dir / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    try:  # best effort: keep the private key readable only by the owner
        key_path.chmod(0o600)
    except OSError:
        pass
    return cert_path, key_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a self-signed TLS cert")
    ap.add_argument("--out-dir", default="output/tls")
    ap.add_argument("--host", action="append", default=None,
                    help="host/IP to put in the cert SAN (repeatable; default: localhost + 127.0.0.1)")
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()
    hosts = args.host or ["localhost", "127.0.0.1"]
    cert_path, key_path = make_cert(args.out_dir, hosts, args.days)
    print(f"Certificate: {cert_path}")
    print(f"Private key: {key_path}")
    print("Serve with:")
    print(f"  python scripts/serve.py --tls-cert {cert_path} --tls-key {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
