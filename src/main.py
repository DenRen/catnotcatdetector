from git import Repo

from .model import HyperParams, infer, train


def run_train_or_infer(is_train: bool, config: HyperParams) -> None:
    repo = Repo(search_parent_directories=False)
    config.git_commit_id = repo.head.object.hexsha
    train(config) if is_train else infer(config)
