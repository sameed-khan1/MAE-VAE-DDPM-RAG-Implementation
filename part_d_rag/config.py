
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"

BIOASQ_DATASET = "enelpol/rag-mini-bioasq"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 384

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
TOP_K = 5
MAX_NEW_TOKENS = 256

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
