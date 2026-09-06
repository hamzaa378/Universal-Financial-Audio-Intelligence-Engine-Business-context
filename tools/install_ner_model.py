from decision_ai.ner_engine import install_model, status

if __name__ == '__main__':
    print('Downloading optional ONNX privacy NER model...')
    path=install_model()
    print('Downloaded to:',path)
    print(status())
