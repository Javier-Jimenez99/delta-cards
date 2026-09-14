"""Card Text Service (Backward compatibility alias for CandidateOCRCache).

Consolidated into CandidateOCRCache to eliminate redundant layer.
"""

from .card_ocr_cache import CandidateOCRCache

# Backward compatibility alias
CardTextService = CandidateOCRCache

__all__ = ["CardTextService", "CandidateOCRCache"]
