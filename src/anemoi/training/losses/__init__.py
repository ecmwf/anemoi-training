# (C) 2023 Your Company Name. All rights reserved.
from .mse import WeightedMSELoss
from .mae import WeightedMAELoss
from .kcrps import KernelCRPS, GroupedMultivariateKernelCRPS, MultivariateKernelCRPS
from .energy import EnergyScore, GroupedEnergyScore
from .divergences import KLDivergenceLoss, RenyiDivergenceLoss
from .vae import VAELoss
from .variogram import VariogramScore
from .spectra import SpectralEnergyLoss, SHTAmplitudePhaseLoss, SHTComplexBetaLoss
from .ignorance import IgnoranceScore
from .composite import CompositeLoss
from .spread_skill import SpreadLoss, SpreadSkillLoss, ZeroSpreadRateLoss

# NOTE: eventually consider moving this to a seperate repository anemoi-losses

__all__ = [
    "CompositeLoss",
    "EnergyScore",
    "GroupedEnergyScore",
    "KernelCRPS",
    "MultivariateKernelCRPS",
    "GroupedMultivariateKernelCRPS",
    "IgnoranceScore",
    "KLDivergenceLoss",
    "RenyiDivergenceLoss",
    "SpectralEnergyLoss",
    "SHTAmplitudePhaseLoss",
    "SHTComplexBetaLoss",
    "SpreadLoss",
    "SpreadSkillLoss",
    "VAELoss",
    "VariogramScore",
    "WeightedMAELoss",
    "WeightedMSELoss",
    "ZeroSpreadRateLoss",
]
