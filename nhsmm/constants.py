# nhsmm/constants.py
import torch
import logging

EPS = 1e-12
MAX_LOGITS = 50.0
DTYPE = torch.float32

# -------------------------
# Logger
# -------------------------
logger = logging.getLogger("nhsmm")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(name)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class HSMMError(ValueError):
    """Custom error class for HSMM module."""
    pass
