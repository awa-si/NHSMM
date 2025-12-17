# nhsmm/constants.py
import torch
import logging

EPS = 1e-12
MAX_LOGITS = 1e4
DTYPE = torch.float32
NEG_INF = torch.finfo(DTYPE).min
DEBUG = False

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
