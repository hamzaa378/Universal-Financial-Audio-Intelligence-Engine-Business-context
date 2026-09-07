"""Install the optional ONNX privacy NER model.

This helper is intentionally runnable both as:
    python tools/install_ner_model.py
and from other working directories.  Add the project root to sys.path before
importing the local decision_ai package.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_ai.ner_engine import install_model, status

if __name__ == '__main__':
    print('Downloading optional ONNX privacy NER model...')
    path = install_model()
    print('Downloaded to:', path)
    print(status())
