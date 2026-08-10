#!/usr/bin/env bash
# Self-signed certs for the BHAIRAV nginx TLS terminator.
# For a public deployment, replace with real certs (Let's Encrypt / your CA)
# and mount them at deploy/certs/.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$DIR/certs"
openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
  -keyout "$DIR/certs/bhairav.key" \
  -out "$DIR/certs/bhairav.crt" \
  -subj "/CN=bhairav.local" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
echo "wrote $DIR/certs/bhairav.{crt,key}"
