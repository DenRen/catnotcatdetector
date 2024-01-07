from dataclasses import dataclass


@dataclass
class Data:
    path: str
    seed: int
    size_h: int
    size_w: int


@dataclass
class Model:
    backbone: str
    backbone_weights: str
    best_model_paht: str


@dataclass
class Training:
    batch_size: int
    epochs: int
    learning_rate: float
    # optimizer: str
    save_top_k: int


@dataclass
class Params:
    data: Data
    model: Model
    training: Training
