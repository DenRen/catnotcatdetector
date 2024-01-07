import hydra
from hydra.core.config_store import ConfigStore

from src.config import Params
from src.model import infer

config_store = ConfigStore.instance()
config_store.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3.2")
def main(config: Params) -> None:
    infer(config)


if __name__ == "__main__":
    main()
