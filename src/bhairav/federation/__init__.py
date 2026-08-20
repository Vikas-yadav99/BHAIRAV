"""Phase 13.3 - Multi-Site Federation: cross-server alert/analytics sharing."""
from .client import FederationClient
from .protocol import FederationMessage

__all__ = ["FederationClient", "FederationMessage"]
