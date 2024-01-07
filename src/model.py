import shutil
from os import cpu_count
from os.path import dirname
from pathlib import Path
from typing import Any

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import get_model

from .config import Params as HyperParams
from .dataset import DatasetSerializedTwoClassImages


class CatDogModel(nn.Module):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = get_model(params.backbone, weights=params.backbone_weights)

        for param in self.backbone.parameters():
            param.requires_grad_(False)

        num_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_feat, 2)

    def forward(self, batch_imgs):
        return self.backbone(batch_imgs)


class CatDogTrainingModule(pl.LightningModule):
    def __init__(self, config: HyperParams, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.training_config = config.training
        self.model = CatDogModel(config.model)
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
        cfg = self.training_config
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.learning_rate)

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


def get_dls(config: HyperParams, is_test: bool = False):
    resize_shape = (config.data.size_h, config.data.size_w)
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    # maybe not need resize?)
    transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resize_shape, antialias=True),
            transforms.Normalize(image_mean, image_std),
        ]
    )

    num_workers = cpu_count()

    def get_dataloader(dir_name: str, shuffle: bool):
        dataset = DatasetSerializedTwoClassImages(
            Path(config.data.path) / dir_name, transformer
        )
        return DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
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


def make_parent_dir(path) -> None:
    parent_dir = Path(dirname(path))
    parent_dir.mkdir(parents=True, exist_ok=True)


def train(config: HyperParams):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=config.training.save_top_k,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        log_every_n_steps=1,
        num_sanity_val_steps=3,
        logger=get_csv_logger(),
        callbacks=[checkpoint_callback],
    )
    training_module = CatDogTrainingModule(config)
    train_dl, val_dl = get_dls(config)

    trainer.fit(training_module, train_dl, val_dl)

    model_path = Path(config.model.best_model_paht)
    make_parent_dir(model_path)
    shutil.copyfile(checkpoint_callback.best_model_path, model_path)


def infer(config: HyperParams):
    trainer = pl.Trainer(
        logger=get_csv_logger(),
        enable_checkpointing=False,
        log_every_n_steps=1,
    )
    training_module = CatDogTrainingModule(config)
    test_dl = get_dls(config, is_test=True)

    model_path = Path(config.model.best_model_paht)
    if not model_path.exists():
        raise RuntimeError(
            f"Model {model_path.absolute()} not found, trying to get last model from logs"
        )

    trainer.test(training_module, test_dl, ckpt_path=model_path)
