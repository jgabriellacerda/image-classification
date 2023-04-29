from pathlib import Path
import torch  # type: ignore

from network import NeuralNetwork


class ImageClassifier:

    def __init__(self, model_path: Path) -> None:
        self.model = NeuralNetwork()
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path))
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

    CLASSES = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    def predict(self, x: torch.Tensor, y: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            pred: torch.Tensor = self.model(x)
            predicted, actual = self.CLASSES[pred[0].argmax(0)], self.CLASSES[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
