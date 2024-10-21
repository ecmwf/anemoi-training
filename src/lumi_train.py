from hydra import compose, initialize
from anemoi.training.train.train import AnemoiTrainer

with initialize(version_base=None, config_path="anemoi/training/config"):
    config = compose(config_name="stretched_grid")

T = AnemoiTrainer(config)

T.train()