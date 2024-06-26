from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from audio_encoder.audio_processing import random_crop, random_mask, random_multiply
from audio_encoder.encoder import Cola


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len=100, augment=True):
        self.data = data
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path = self.data[idx]

        x = np.load(npy_path)

        if self.augment:
            x = random_mask(x)

        x1 = random_crop(x, crop_size=self.max_len)
        x2 = random_crop(x, crop_size=self.max_len)

        if self.augment:
            x1 = random_multiply(x1)
            x2 = random_multiply(x2)

        x1 = torch.tensor(x1, dtype=torch.float)
        x2 = torch.tensor(x2, dtype=torch.float)

        return x1, x2


class DecayLearningRate(pl.Callback):
    def __init__(self):
        self.old_lrs = []

    def on_train_start(self, trainer, pl_module):
        # track the initial learning rates
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            group = []
            for param_group in optimizer.param_groups:
                group.append(param_group["lr"])
            self.old_lrs.append(group)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            old_lr_group = self.old_lrs[opt_idx]
            new_lr_group = []
            for p_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = old_lr_group[p_idx]
                new_lr = old_lr * 0.99
                new_lr_group.append(new_lr)
                param_group["lr"] = new_lr
            self.old_lrs[opt_idx] = new_lr_group


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--mp3_path")
    args = parser.parse_args()

    mp3_path = Path(args.mp3_path)

    batch_size = 128
    epochs = 512

    files = sorted(list(glob(str(mp3_path / "*/*.npy"))))

    _train, test = train_test_split(files, test_size=0.05, random_state=1337)

    train, val = train_test_split(_train, test_size=0.05, random_state=1337)

    train_data = AudioDataset(train, augment=True)
    test_data = AudioDataset(test, augment=False)
    val_data = AudioDataset(val, augment=False)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=8, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=8, shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=8
    )

    model = Cola()

    logger = TensorBoardLogger(
        save_dir=".",
        name="lightning_logs",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_acc", mode="max", filepath="models/", prefix="encoder"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[DecayLearningRate()],
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(test_dataloaders=test_loader)
