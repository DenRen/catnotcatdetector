import os
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from shutil import unpack_archive

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from .datautils import ContinuousImageArray
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


# @deprecated
def download_data():
    data_dir = HyperParams().data_dir
    zip_name = "data.zip"
    zip_path = data_dir / zip_name

    if not zip_path.exists():
        file_link = f"https://www.dropbox.com/s/gqdo90vhli893e0/{zip_name}"
        # Make dropbox file link downloadable
        file_link += "?dl=1"

        # download(file_link, str(zip_path))

    unpack_archive(zip_path, data_dir)


def get_torch_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Dataset special for cats and dogs
class DatasetSerializedImages(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root_dir: str,
        transform: None,
        num_classes: int = 2,
    ) -> None:
        def two_class_exc(true_cond: bool = False) -> None:
            if not true_cond:
                raise NotImplementedError("DatasetSerializedImages only for 2 class")

        two_class_exc(num_classes == 2)
        super().__init__(root_dir, transform=transform)

        class_names = sorted(os.listdir(root_dir))
        two_class_exc(len(class_names) == 2)

        self.datasets = []
        for class_name in class_names:
            class_path = join(root_dir, class_name)
            img_arr_path = join(class_path, f"{class_name}_bin")
            pos_arr_path = join(class_path, f"{class_name}_pos.npy")
            cia = ContinuousImageArray(img_arr_path, pos_arr_path)
            self.datasets.append(cia)

        self.size_first_class = len(self.datasets[0])

    def __getitem__(self, idx: int):
        ds_idx = int(idx >= self.size_first_class)
        idx -= ds_idx * self.size_first_class

        img = self.datasets[ds_idx][idx].copy()
        if self.transform is not None:
            img = self.transform(img)

        return img, ds_idx

    def __len__(self) -> int:
        return self.size_first_class + len(self.datasets[1])


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

        # load dataset using torchvision.datasets.ImageFolder
        # train_dataset = torchvision.datasets.ImageFolder(
        #     hp.data_dir / "train_11k", transform=self.transformer
        # )
        # test_dataset = torchvision.datasets.ImageFolder(
        #     hp.data_dir / "test_labeled", transform=self.transformer
        # )

        train_dataset = DatasetSerializedImages(
            hp.data_dir / "train_11k", self.transformer
        )
        test_dataset = DatasetSerializedImages(
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
        val_dataset = DatasetSerializedImages(
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
