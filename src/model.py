from dataclasses import dataclass
from os import listdir
from pathlib import Path
from typing import Any

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import DatasetSerializedTwoClassImages


@dataclass
class HyperParams:
    data_dir: Path = Path("serialized_data")
    num_workers: int = 4
    size_h: int = 96
    size_w: int = 96
    num_classes: int = 2
    epoch_num: int = 2
    batch_size: int = 256
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    embedding_size: int = 128
    model_path: str = data_dir / "model.ckpt"


class CatDogModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.backbone = torchvision.models.resnet18(weights=weights)

        for param in self.backbone.parameters():
            param.requires_grad_(False)

        num_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_feat, 2)

    def forward(self, batch_imgs):
        return self.backbone(batch_imgs)


class CatDogTrainingModule(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.model = CatDogModel()
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs, class_idxs = batch

        pred_class_idxs = self.model(imgs)
        loss = self.loss(pred_class_idxs, class_idxs)

        pred = F.softmax(pred_class_idxs, dim=1).argmax(dim=1)
        acc = self.accuracy(pred, class_idxs)

        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs, class_idxs = batch

        pred_class_idxs = self.model(imgs)
        loss = self.loss(pred_class_idxs, class_idxs)

        pred = F.softmax(pred_class_idxs, dim=1).argmax(dim=1)
        acc = self.accuracy(pred, class_idxs)

        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs, class_idxs = batch

        pred_class_idxs = self.model(imgs)
        loss = self.loss(pred_class_idxs, class_idxs)

        pred = F.softmax(pred_class_idxs, dim=1).argmax(dim=1)
        acc = self.accuracy(pred, class_idxs)

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=5e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.2,
            patience=5,
            verbose=True,
        )

        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        return [opt], [lr_dict]


def get_dls(hp, is_test: bool = False):
    transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((hp.size_h, hp.size_w), antialias=True),
            transforms.Normalize(hp.image_mean, hp.image_std),
        ]
    )

    def get_dataloader(dir_name: str, shuffle: bool):
        dataset = DatasetSerializedTwoClassImages(hp.data_dir / dir_name, transformer)
        return DataLoader(
            dataset,
            batch_size=hp.batch_size,
            shuffle=shuffle,
            num_workers=hp.num_workers,
        )

    if is_test:
        test_dl = get_dataloader("test_labeled", shuffle=False)
        return test_dl
    else:
        train_dl = get_dataloader("train_11k", shuffle=True)
        val_dl = get_dataloader("val", shuffle=False)
        return train_dl, val_dl


def get_csv_logger():
    return CSVLogger("logs")


def train():
    hp = HyperParams()
    trainer = pl.Trainer(
        max_epochs=hp.epoch_num,
        log_every_n_steps=1,
        num_sanity_val_steps=3,
        logger=get_csv_logger(),
    )
    training_module = CatDogTrainingModule()
    train_dl, val_dl = get_dls(hp)

    trainer.fit(training_module, train_dl, val_dl)


def infer():
    trainer = pl.Trainer(
        logger=get_csv_logger(),
    )
    training_module = CatDogTrainingModule()
    test_dl = get_dls(HyperParams(), is_test=True)

    pl_log_dir = Path("logs") / "lightning_logs"
    last_version_path = pl_log_dir / sorted(listdir(pl_log_dir))[-1]
    chekpoints_dir = last_version_path / "checkpoints"
    chekpoint_path = chekpoints_dir / listdir(chekpoints_dir)[0]

    trainer.test(training_module, test_dl, ckpt_path=chekpoint_path)
