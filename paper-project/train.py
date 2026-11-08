import transformers
import torch
import time


def train_model():
    DEVICE = "mps"
    config = transformers.Config(num_epochs = 1000, device=torch.device(DEVICE))
    print(f"device: {config.device}")

    start = time.perf_counter()
    metrics = transformers.train_model(config)
    end = time.perf_counter()
    print(f"Время выполнения: {end - start:.4f} секунд")


if __name__ == "__main__":
    train_model()