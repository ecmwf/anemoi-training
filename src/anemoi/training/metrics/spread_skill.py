
from typing import Optional
from torch import Tensor


class SpreadSkillMetric(nn.Module):
    def __init__(
        self,
        area_weights: Tensor,
        feature_weights: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        """SpreadSkill.

        # This presents the micro spread skill. In this formulation the spread skill is calculated
        # for each ensemble prediction and then averaged

        Parameters
        ----------
        area_weights : Tensor
            Weights by area
        data_variances : Optional[Tensor], optional
            precomputed, per-variable stepwise variance estimate, by default None
        """
        super().__init__(area_weights, feature_weights)

    def forward(
        self,
        preds: Tensor,
        target: Tensor,
        squash: bool = True,

        indices_adjusted=None,
        **kwargs,
    ) -> Tensor:
        # Logic Type: Pre-operation filtering, Post operation variable scaling, Post operation LatLon Scaling
        # In this case, preoperation filtering is applied since it leads to much quicker loops e.g. smaller tensors

        # assert enum_feature_weight_scalenorm in [0, 1, 2]
        # Type 0: Use the weights in feature_weight_scales value
        # Type 1: Ignore feature_weight_scales value and use uniform weights
        # Type 2: No scaling

        preds, target, indices = feature_indexing(
            preds, target, indices_adjusted, self.feature_ignored_flag, self.scale_binarized_indices
        )

        bs, m, latlon, feat = preds.shape

        rmse = torch.sqrt(torch.square(preds.mean(dim=1) - target))  # shape (bs, latlon, nvar)

        rmse = feature_weight_scale_and_norm(
            rmse,
            enum_feature_weight_scalenorm,
            self.feature_weight_scale_norm_constant,
            indices,
        )
        rmse = area_scale_and_norm(
            rmse,
            0,  # scale_and_norm weighted by area
            self.area_weights_scale_and_norm_constant,
        )

        ens_stdev = torch.sqrt(torch.square(preds - preds.mean(dim=1, keepdim=True)).sum(dim=1) / (m - 1))

        ens_stdev = feature_weight_scale_and_norm(
            ens_stdev,
            enum_feature_weight_scalenorm,
            self.feature_weight_scale_norm_constant,
            indices,
        )
        ens_stdev = area_scale_and_norm(
            ens_stdev,
            enum_area_weight_scalenorm,
            self.area_weights_scale_and_norm_constant,
        )  # shape (bs, ens, latlon, nvar)

        if squash:
            rmse = rmse.sum()  # shape (bs, latlon, nvar)
            ens_stdev = ens_stdev.sum()

            return ens_stdev / rmse
        else:
            rmse = rmse.sum(dim=0)  # shape (latlon, nvar)
            ens_stdev = ens_stdev.sum(dim=0)

            return ens_stdev / rmse

    @cached_property
    def log_name(self):
        return "spread_skill"


class SpreadMetric(Loss):
    def __init__(
        self,
        area_weights: Tensor,
        feature_weights: Optional[Tensor] = None,
        power: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(area_weights, feature_weights)
        """Spread.

        Parameters
        ----------
        area_weights : Tensor
            Weights by area
        data_variances : Optional[Tensor], optional
            precomputed, per-variable stepwise variance estimate, by default None
        """
        super().__init__(area_weights, feature_weights)
        self.power = power

    def forward(
        self,
        preds: Tensor,
        target: Tensor,
        squash: bool = True,
        enum_feature_weight_scalenorm: int = 0,
        enum_area_weight_scalenorm: int = 0,
        indices_adjusted=None,
        deterministic=True,
        **kwargs,
    ) -> Tensor:
        preds, target, indices = feature_indexing(
            preds, target, indices_adjusted, self.feature_ignored_flag, self.scale_binarized_indices
        )

        spread = torch.pow(
            torch.pow(preds - preds.mean(dim=1, keepdim=True), self.power).mean(dim=1), 1 / self.power
        )  # shape (bs, latlon, nvar)

        spread = feature_weight_scale_and_norm(
            spread, enum_feature_weight_scalenorm, self.feature_weight_scale_norm_constant, self.scale_binarized_indices
        )

        spread = area_scale_and_norm(spread, enum_area_weight_scalenorm, self.area_weights_scale_and_norm_constant)

        if squash:
            return spread.sum() / spread.shape[0]  # shape (1)
        else:
            return spread.mean(dim=0)  # shape ( latlon, nvar)

    @cached_property
    def log_name(self):
        return f"spread_p{self.power:.1f}"


class ZeroSpreadRateMetric(Loss):
    def __init__(
        self,
        area_weights: Tensor,
        feature_weights: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        """Spread.

        Parameters
        ----------
        """
        super().__init__(area_weights, feature_weights)

    def forward(
        self,
        preds: Tensor,
        target: Tensor,
        squash: bool = True,
        enum_feature_weight_scalenorm: int = 0,
        enum_area_weight_scalenorm: int = 0,
        indices_adjusted=None,
        deterministic=True,
        **kwargs,
    ) -> Tensor:
        assert enum_feature_weight_scalenorm == 0, "Feature weight scaling not supported for ZeroSpreadRateMetric"
        assert enum_area_weight_scalenorm == 0, "Area weight scaling not supported for ZeroSpreadRateMetric"

        preds, target, indices = feature_indexing(
            preds, target, indices_adjusted, self.feature_ignored_flag, self.scale_binarized_indices
        )

        spread = (torch.square(preds - preds.mean(dim=1, keepdim=True))).mean(dim=1)  # shape (bs, latlon, nvar)

        spread_occurence = torch.where(spread == 0, 1.0, 0.0)

        spread_occurence = feature_weight_scale_and_norm(spread_occurence, 2, self.feature_weight_scale_norm_constant, indices)

        spread_occurence = area_scale_and_norm(
            spread_occurence,
            2,
            self.area_weights_scale_and_norm_constant,
        )

        if squash:
            return spread_occurence.mean()  # shape (1)

        else:
            return spread.mean(dim=(0, 1))  # shape ( latlon, nvar)

    @cached_property
    def log_name(self):
        return "zero_spread_rate"
