"""Pipeline module exporting OmniProcessor and auto-detection tools."""
from .omni_processor import OmniProcessor, detect_modality

__all__ = ["OmniProcessor", "detect_modality"]
