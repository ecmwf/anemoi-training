#!/bin/env python3
from anemoi.datasets import open_dataset
from anemoi.utils.data_structures import NestedTrainingSample, TorchNestedAnemoiTensor
import torch
from torch import nn
from pathlib import Path


def get_dataset():
    HERE = Path(__file__).parent
    path = HERE / ".." / "src" / "anemoi" / "training" / "config" / "dataloader" / "observations.yaml"
    with path.open("r") as f:
        import yaml

        cfg = yaml.safe_load(f)
    cfg = cfg["dataset"]
    return open_dataset(cfg)

class DummyEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleDict()
        self.encoders["era5"] = nn.Linear(101, 64)
        self.encoders["metar"] = nn.Linear(14, 64)
        self.encoders["noaa-atms"] = nn.Linear(32, 64)

        self.mixer = nn.Linear(64, 64)

    def forward(self, x: dict) -> dict:
        y = {}

        assert set(x.keys()) == set(self.encoders.keys()), f"Keys do not match: {set(x.keys())} != {set(self.encoders.keys())}"

        for key in self.encoders.keys():
            encoder, xt = self.encoders[key], x[key]
            xt = torch.from_numpy(xt)
            y[key] = encoder(xt)

        # return y
        y_as_list = [y[key] for key in self.encoders.keys()]
        return self.mixer(torch.cat(y_as_list))


dummy_model = DummyEncoderModel()
dummy_model.train()

def get_data(i):
    ds = get_dataset()
    i_s = [i, i + 1, i + 2, i + 3]
    print()
    print(f"-> Using data for {ds.dates[i_s[0]]} to {ds.dates[i_s[-1]]}")
    return [ds[_] for _ in i_s]

i = 27
data = get_data(i)
assert len(data) == 4  # 4 states
assert len(data[0]) == 3 # era5 + 2 satellites
assert len(data[1]) == 3 # era5 + 2 satellites

x = data[0]
y_ref = data[1]
print(x)

# x = OrderedDict()
# x["seviri"] = torch.from_numpy(data[0].squeeze(axis=1).T)
# x["metar"] = torch.from_numpy(data[1].squeeze(axis=1).T)
# x["noaa-atms"] = torch.from_numpy(data[2].squeeze(axis=1).T)
# x["era"] = torch.from_numpy(data[3].squeeze(axis=1).T)

y = dummy_model(x)

print(f"Model input shapes: {[obs + ': ' + str(list(x_in.shape)) for (obs, x_in) in x.items()]}")
print(f"Model output shape: {list(y.shape)}")
assert y.shape == (sum([_.shape[-1] for i,_ in x.items()]), 64) 
