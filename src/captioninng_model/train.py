import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  

import torch
import joblib
from torch.utils.data import DataLoader
from data.flickr_dataset import Flickr8k          
from src.constants import CONFIG_PATH
from src.utils.common import read_yaml
from src.captioninng_model.get_gpt2_decoder import Model


def train(path=CONFIG_PATH):
    config = read_yaml(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataloader = DataLoader(
        Flickr8k(path),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = Model(path).to(device)                 
    model.train()

    optimizer = torch.optim.AdamW([
        {"params": model.projection.parameters(), "lr": 1e-3},
        {"params": model.gpt_model.parameters(), "lr": 5e-5}
    ])

    for epoch in range(config.training.epochs):
        total_loss = 0

        for batch_idx, (features, captions) in enumerate(dataloader):
            features = features.to(device)

            optimizer.zero_grad()
            loss = model(features, captions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

    joblib.dump({
        "projection": model.projection.state_dict(),
        "gpt_model": model.gpt_model.state_dict()
    }, config.training.save_path)
    print(f"Model saved at {config.training.save_path}")


if __name__ == "__main__":                         # fix 3 — correct syntax
    train()