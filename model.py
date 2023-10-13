import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from tqdm.auto import tqdm


class Runner:
    """Runner for experiments with supervised model."""

    def __init__(self, model, opt, device, checkpoint_name=None):
        self.model = model
        self.opt = opt
        self.device = device
        self.checkpoint_name = checkpoint_name

        self.epoch = 0
        self.output = None
        self.metrics = None
        self._global_step = 0
        self._set_events()
        self._top_val_accuracy = -1
        self.log_dict = {"train": [], "val": [], "test": []}

        if model is not None:
            model = model.to(self.device)

    def _set_events(self):
        """
        Additional method to initialize variables, which may store logging and
        evaluation info. The implementation below is extremely simple and only
        provided to help monitor performance.
        """
        self._phase_name = ""
        self.events = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list),
        }

    def _reset_events(self, event_name):
        self.events[event_name] = defaultdict(list)

    def forward(self, img_batch, **kwargs):
        """
        Forward method for your Runner.
        Should not be called directly outside your Runner.
        In simple case, this method should only implement your model forward
        pass.
        It should also return the model predictions and/or other meta info.

        Args:
            batch (mapping[str, Any]): dictionary with data batches from
                DataLoader.
            **kwargs: additional parameters to pass to the model.
        """
        logits = self.model(img_batch)
        output = {
            "logits": logits,
        }
        return output

    def run_criterion(self, batch):
        """
        Applies the criterion to the data batch and the model output, saved in
        self.output.

        Args:
            batch (mapping[str, Any]): dictionary with data batches from
                DataLoader.
        """
        raise NotImplementedError("To be implemented")

    def output_log(self):
        """
        Output log using the statistics collected in
            self.events[self._phase_name].
        Implement this method for logging purposes.
        """
        raise NotImplementedError("To be implemented")

    def _run_batch(self, batch):
        """
        Runs batch of data through the model, performing forward pass.
        This implementation performs data passing to necessary device and is
        adapted to the default pyTorch DataLoader.

        Args:
            batch (mapping[str, Any]): dictionary with data batches from
                DataLoader.
        """
        # split batch tuple into data batch and label batch
        X_batch, y_batch = batch

        # update the global step in iterations over source data
        self._global_step += len(y_batch)

        # move data to target device
        X_batch = X_batch.to(self.device)

        # run the batch through the model
        self.output = self.forward(X_batch)

    def _run_epoch(self, loader, train_phase=True, output_log=False, **kwargs):
        """
        Method that runs one epoch of the training process.

        Args:
            loader (DataLoader): data loader to iterate
            train_phase (bool): boolean value to determine if this is the
                training phase. Changes behavior for dropout, batch normalization, etc.
        """
        # Train phase
        # enable or disable dropout / batch_norm training behavior
        self.model.train(train_phase)

        _phase_description = "Training" if train_phase else "Evaluation"
        for batch in tqdm(loader, desc=_phase_description, leave=False):
            # forward pass through the model using preset device
            self._run_batch(batch)

            # train on batch: compute loss and gradients
            with torch.set_grad_enabled(train_phase):
                loss = self.run_criterion(batch)

            # compute backward pass if training phase
            # reminder: don't forget the optimizer step and zeroing the grads
            if train_phase:
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

        self.log_dict[self._phase_name].append(
            np.mean(self.events[self._phase_name]["loss"])
        )

        if output_log:
            self.output_log(**kwargs)

    def train(self, train_loader, val_loader, n_epochs, model=None, opt=None, **kwargs):
        """
        Training process method, that runs for n_epochs over train_loader and
        performs validation using val_loader.

        Args:
            train_loader (DataLoader): training set data loader to iterate over
            val_loader (DataLoader): validation set data loader to iterate over
            n_epochs (int): epoch number to train for
            model (Model): torch nn.Module or nested class, that implements the
                model. Overwrites self.model.
            opt (Optimizer): torch optimizer to be used for loss minimization.
                Overwrites self.opt.
            **kwargs: additional parameters to pass to self.validate.
        """

        for _epoch in range(n_epochs):
            start_time = time.time()
            self.epoch += 1
            print(f"epoch {self.epoch:3d}/{n_epochs:3d} started")

            # training part
            self._set_events()
            self._phase_name = "train"
            self._run_epoch(train_loader, train_phase=True)

            dt = time.time() - start_time
            print(f"epoch {self.epoch:3d}/{n_epochs:3d} took {dt:.2f}s")

            # validation part
            self._phase_name = "val"
            self.validate(val_loader, **kwargs)
            self.save_checkpoint()

    @torch.no_grad()  # we do not need to save gradients during validation
    def validate(self, loader, model=None, phase_name="val", **kwargs):
        """
        Validation process method, that estimates the performance of
        self.model on validation data in loader.

        Args:
            loader (DataLoader): validation set data loader to iterate over
            model (Model): torch nn.Module or nested class, that implements the
                model. Overwrites self.model.
            opt (Optimizer): torch optimizer to be used for loss minimization.
                Overwrites self.opt.
            **kwargs: additional parameters to pass to self.validate.
        """
        self._phase_name = phase_name
        self._reset_events(phase_name)
        self._run_epoch(loader, train_phase=False, output_log=True, **kwargs)
        return self.metrics


class CNNRunner(Runner):
    def run_criterion(self, batch):
        """
        Applies the criterion to the data batch and the model output, saved in
        self.output.

        Args:
            batch (mapping[str, Any]): dictionary with data batches from
                DataLoader.
        """
        X_batch, label_batch = batch
        label_batch = label_batch.to(self.device)

        logit_batch = self.output["logits"]

        # compute loss funciton
        loss = nn.CrossEntropyLoss()(logit_batch, label_batch)

        scores = F.softmax(logit_batch, 1).detach().cpu().numpy()[:, 1]
        scores = scores.tolist()
        labels = label_batch.detach().cpu().numpy().ravel().tolist()

        # log some info
        np_loss = loss.detach().cpu().numpy()
        self.events[self._phase_name]["loss"].append(np_loss)
        self.events[self._phase_name]["scores"].extend(scores)
        self.events[self._phase_name]["labels"].extend(labels)

        return loss

    def save_checkpoint(self):
        val_accuracy = self.metrics["accuracy"]
        # save checkpoint of the best model to disk
        if val_accuracy > self._top_val_accuracy and self.checkpoint_name is not None:
            self._top_val_accuracy = val_accuracy
            torch.save(self.model, open(self.checkpoint_name, "wb"))

    def output_log(self, **kwargs):
        """
        Output log using the statistics collected in
        self.events[self._phase_name].
        Let's have a fancy code for classification metrics calculation.
        """
        scores = np.array(self.events[self._phase_name]["scores"])
        labels = np.array(self.events[self._phase_name]["labels"])

        assert len(labels) > 0, print("Label list is empty")
        assert len(scores) > 0, print("Score list is empty")
        assert len(labels) == len(scores), print(
            "Label and score lists are of different size"
        )

        self.metrics = {
            "loss": np.mean(self.events[self._phase_name]["loss"]),
            "accuracy": accuracy_score(labels, np.int32(scores > 0.5)),
            "f1": f1_score(labels, np.int32(scores > 0.5)),
        }
        print(f"{self._phase_name}: ", end="")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()]))

        # self.save_checkpoint()

        visualize = kwargs.get("visualize", False)
        if visualize:
            # tensorboard for the poor
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            ax1.plot(self.log_dict["train"], color="b", label="train")
            ax1.plot(self.log_dict["val"], color="c", label="val")
            ax1.legend()
            ax1.set_title("Train/val loss.")

            class_0_scores = np.array(scores)[np.array(labels) == 0]
            class_1_scores = np.array(scores)[np.array(labels) == 1]
            ax2.hist(
                class_0_scores,
                bins=50,
                range=[0, 1.01],
                color="r",
                alpha=0.7,
                label="cats",
            )
            ax2.hist(
                class_1_scores,
                bins=50,
                range=[0, 1.01],
                color="g",
                alpha=0.7,
                label="dogs",
            )
            ax2.legend()
            ax2.set_title("Validation set score distribution.")

            plt.show()

        csv_file_name = kwargs.get("dump_csv", "")
        if csv_file_name:
            compare = np.array([scores, labels]).T
            np.savetxt(csv_file_name, compare, delimiter=",")


@dataclass
class HyperParams:
    data_path: Path = Path("data")
    num_workers: int = 4
    size_h: int = 96
    size_w: int = 96
    num_classes: int = 2
    epoch_num: int = 2
    batch_size: int = 256
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    embedding_size: int = 128
    model_file_name: str = "model.ckpt"


def get_torch_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ModelWrapper:
    def __init__(self):
        hp = self.hp = HyperParams()
        self.device = get_torch_device()
        print(f"Selected device: {self.device}")

        self.transformer = transforms.Compose(
            [
                transforms.Resize((hp.size_h, hp.size_w)),
                transforms.ToTensor(),
                transforms.Normalize(hp.image_mean, hp.image_std),
            ]
        )

    def train(self):
        hp = self.hp

        # load dataset using torchvision.datasets.ImageFolder
        train_dataset = torchvision.datasets.ImageFolder(
            hp.data_path / "train_11k", transform=self.transformer
        )
        test_dataset = torchvision.datasets.ImageFolder(
            hp.data_path / "test_labeled", transform=self.transformer
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

        runner = CNNRunner(model_resnet18, opt_resnet, self.device, hp.model_file_name)
        runner.train(
            train_batch_gen, test_batch_gen, n_epochs=hp.epoch_num, visualize=False
        )

    def evaluate_model(self):
        hp = self.hp
        model = torch.load(open(hp.model_file_name, "rb"))
        print(f"Model {hp.model_file_name} is loaded")

        runner = CNNRunner(model, None, self.device, hp.model_file_name)

        # load test data also, to be used for final evaluation
        val_dataset = torchvision.datasets.ImageFolder(
            hp.data_path / "val", transform=self.transformer
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
