from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_ai.ner_engine import install_model, status

if __name__ == '__main__':
    print('Downloading optional ONNX privacy NER model...')
    path=install_model()
    print('Downloaded to:',path)
    print(status())
