import transformers

def train_model():
    config = transformers.Config(num_epochs = 1000)
    metrics = transformers.train_model(config)
    print(metrics)


if __name__ == "__main__":
    train_model()