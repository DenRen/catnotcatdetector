from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from .dataset import DatasetSerializedTwoClassImages
from .runner import CNNRunner


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


def get_torch_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ModelWrapper:
    def __init__(self):
        hp = self.hp = HyperParams()
        self.device = get_torch_device()
        print(f"Selected device: {self.device}")

        self.transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((hp.size_h, hp.size_w), antialias=True),
                transforms.Normalize(hp.image_mean, hp.image_std),
            ]
        )

    def train(self):
        hp = self.hp

        train_dataset = DatasetSerializedTwoClassImages(
            hp.data_dir / "train_11k", self.transformer
        )
        test_dataset = DatasetSerializedTwoClassImages(
            hp.data_dir / "test_labeled", self.transformer
        )

        train_batch_gen = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=hp.batch_size,
            shuffle=True,
            num_workers=hp.num_workers,
        )
        test_batch_gen = torch.utils.data.DataLoader(
            test_dataset, batch_size=hp.batch_size, num_workers=hp.num_workers
        )

        ################
        # Fine tunning #
        ################

        # Load pre-trained model
        model_resnet18 = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")

        # Disable gradient updates for all the layers except  the final layer
        for p in model_resnet18.parameters():
            p.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_feat = model_resnet18.fc.in_features

        # add your own prediction part: FC layer for 2 classes
        model_resnet18.fc = nn.Linear(num_feat, hp.num_classes)

        # Use available device for calculations
        model_resnet18 = model_resnet18.to(self.device)

        # Observe that only parameters of final layer are being optimized as opposed to
        # before
        opt_resnet = torch.optim.Adam(model_resnet18.fc.parameters(), lr=1e-3)

        runner = CNNRunner(model_resnet18, opt_resnet, self.device, hp.model_path)
        runner.train(
            train_batch_gen, test_batch_gen, n_epochs=hp.epoch_num, visualize=False
        )

    def evaluate_model(self):
        hp = self.hp
        model = torch.load(open(hp.model_path, "rb"))
        print(f"Model {hp.model_path} is loaded")

        runner = CNNRunner(model, None, self.device, hp.model_path)

        # load test data also, to be used for final evaluation
        val_dataset = DatasetSerializedTwoClassImages(
            hp.data_dir / "val", transform=self.transformer
        )
        val_batch_gen = torch.utils.data.DataLoader(
            val_dataset, batch_size=hp.batch_size, num_workers=hp.num_workers
        )

        runner.validate(val_batch_gen, model, "val", dump_csv="validate_res.csv")


def train():
    model_wrapper = ModelWrapper()
    model_wrapper.train()


def infer():
    model_wrapper = ModelWrapper()
    model_wrapper.evaluate_model()
