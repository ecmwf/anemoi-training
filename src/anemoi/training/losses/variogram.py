
class VariogramScore(Loss):
    # NOTE: Current implementation only calculates the variogram score in the spatial dimension
    # TODO (rilwan-ade): Allow the dimension to be specified e.g. temporal or spatial
    def __init__(
        self,
        area_weights: Tensor,
        feature_weights: Optional[Tensor] = None,
        group_on_dim: int = -2,
        power: torch.Tensor | float = 1.0,
        **kwargs,
    ) -> None:
        """Variogram score for evaluating spatial forecasts.

        Args:
            area_weights: Area weights for spatial dimensions.
            feature_weights: Optional scaling factors to ensure equal contribution from all variables.
            power: The power to which differences are raised. Typically 2 for squared differences.
        """
        super().__init__(
            area_weights,
            feature_weights,
            area_weight_preoperation_exponent=power * 2,
        )
        assert group_on_dim == -2, "Grouping must be done on the spatial dimension"
        self.group_on_dim = group_on_dim

        self.register_buffer("power", torch.as_tensor(power), persistent=False)
        if self.power.ndim == 0:
            self.power = self.power.unsqueeze(0)

    def _calc_variogram_score(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            preds (Tensor): Predicted ensemble, shape (batch_size, ens_size, latlon, n_vars).
            target (Tensor): Target values, shape (batch_size, latlon, n_vars).

        """

        preds_diff = torch.mean(
            torch.abs(preds.unsqueeze(self.group_on_dim) - preds.unsqueeze(self.group_on_dim - 1)).pow(self.power), dim=-4
        )  # (bs,  latlon, latlon, n_vars)
        target_diff = torch.abs(target.unsqueeze(self.group_on_dim) - target.unsqueeze(self.group_on_dim - 1)).pow(
            self.power
        )  # (bs, latlon, latlon, n_vars)

        # Calculate the variogram components
        vario_score = preds_diff - target_diff  # (bs, latlon, latlon, n_vars)
        vario_score = vario_score.pow(2)  # Squaring the difference

        # Averaging over ensemble count
        # vario_score = vario_score.mean(dim=( self.group_on_dim-1, self.group_on_dim))  # (bs, n_vars)
        vario_score = vario_score.sum(dim=(self.group_on_dim - 1, self.group_on_dim))

        return vario_score  # (bs, n_vars)

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
        """Forward pass of the Variogram Score computation.

        Args:
            preds (Tensor): Predicted ensemble, shape (batch_size, ens_size, latlon, n_vars).
            target (Tensor): Target values, shape (batch_size, latlon, n_vars).
            squash (bool, optional): Whether to aggregate scores into a single value.

        Returns:
            Tensor: Variogram score. Scalar if squash is True; otherwise, shape (latlon, n_vars).
        """
        # Calculate pairwise differences between predictions in spatial dimension for each ensemble in the predictions;
        # similarily for target

        # Logic: Post operation variable scaling, PreOperation LatLon Scaling

        preds, target, indices = feature_indexing(
            preds, target, indices_adjusted, self.feature_ignored_flag, self.scale_binarized_indices
        )

        preds_ = area_scale_and_norm(
            preds,
            enum_area_weight_scalenorm,
            self.area_weights_scale_and_norm_constant,
        )
        target_ = area_scale_and_norm(
            target,
            enum_area_weight_scalenorm,
            self.area_weights_scale_and_norm_constant,
        )

        vario_score = self._calc_variogram_score(preds_, target_)  # (bs, n_vars)

        vario_score_ = feature_weight_scale_and_norm(
            vario_score,
            enum_feature_weight_scalenorm,
            self.feature_weight_scale_norm_constant,
            indices,
        )
        vario_score = vario_score_

        if squash:
            return vario_score.sum() / vario_score.shape[0]  # (1)

        # If not squashed, return the weighted score for each location and variable
        return vario_score.mean(dim=0)  # shape (nvars)

    @cached_property
    def log_name(self):
        power_str = "p" + "_".join([str(p) for p in self.power.tolist()])

        return f"vgram_{power_str}"
