from pathlib import Path
import torch  # type: ignore

from network import NeuralNetwork
from predict import ImageClassifier
from train import ImageClassifierTrainer


def main():
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Using {device} device")

    weights_path = Path('model.pth')
    trainer = ImageClassifierTrainer(weights_path)
    trainer.train_epochs(10)
    x, y = trainer.test_data[0][0], trainer.test_data[0][1]
    ImageClassifier(weights_path).predict(x, y)


if __name__ == '__main__':
    main()
