import hydra
from hydra.core.config_store import ConfigStore

from .config import Params
from .model import infer

config_store = ConfigStore.instance()
config_store.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3.2")
def main_infer(config: Params) -> None:
    infer(config)
