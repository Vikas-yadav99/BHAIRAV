"""Phase 20: GDPR/Privacy Compliance — retention, consent, deletion."""
from .retention import RetentionPolicy, RetentionManager
from .consent import ConsentManager, ConsentRecord
from .deletion import DeletionService

__all__ = [
    "RetentionPolicy", "RetentionManager", "ConsentManager",
    "ConsentRecord", "DeletionService",
]
