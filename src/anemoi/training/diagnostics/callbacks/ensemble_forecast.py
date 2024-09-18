
from aifs.diagnostics.plots import plot_spread_skill
from aifs.diagnostics.plots import plot_spread_skill_bins

from aifs.metrics.ranks import RankHistogram
from aifs.metrics.spread import SpreadSkill

class SpreadSkillPlot(PlotCallback):
    def __init__(self, config, val_dset_len, **kwargs):
        super().__init__(config, op_on_batch=True, val_dset_len=val_dset_len, **kwargs)

        self.spread_skill = SpreadSkill(
            rollout=config.diagnostics.metrics.rollout_eval.rollout,
            nvar=len(config.diagnostics.plot.parameters),
            nbins=config.diagnostics.metrics.rollout_eval.num_bins,
            time_step=int(config.data.timestep[:-1]),
        )

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: torch.Tensor, batch_idx: int
    ) -> None:
        if SpreadSkillPlot.flag_plot_on_batch(batch_idx, pl_module.ens_comm_group_rank, self.plot_frequency, self.min_iter_to_plot):
            area_weights = pl_module.graph_data[("era", "to", "era")].area_weights.to(device=pl_module.device)

            plot_parameters_dict = {
                pl_module.data_indices.model.output.name_to_index[name]: name
                for name in self.eval_plot_parameters
                if name in pl_module.data_indices.model.output.name_to_index
            }

            preds_denorm = outputs["preds_denorm"]
            targets_denorm = outputs["targets_denorm"]

            rollout_steps = len(preds_denorm)

            rmse = torch.zeros(
                (rollout_steps, len(self.eval_plot_parameters)),
                dtype=batch[0].dtype,
                device=pl_module.device,
            )
            spread = torch.zeros_like(rmse)
            binned_rmse = torch.zeros(
                (rollout_steps, len(self.eval_plot_parameters), self.spread_skill.nbins - 1),
                dtype=batch[0].dtype,
                device=pl_module.device,
            )
            binned_spread = torch.zeros_like(binned_rmse)

            for rollout_step in range(rollout_steps):
                pred_denorm = preds_denorm[rollout_step]
                target_denorm = targets_denorm[rollout_step]

                for midx, (pidx, _) in enumerate(plot_parameters_dict.items()):
                    (
                        rmse[rollout_step, midx],
                        spread[rollout_step, midx],
                        binned_rmse[rollout_step, midx],
                        binned_spread[rollout_step, midx],
                    ) = self.spread_skill.calculate_spread_skill(pred_denorm, target_denorm, pidx, area_weights)

            _ = self.spread_skill(rmse, spread, binned_rmse, binned_spread, pl_module.device)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        plot_parameters_dict = {
            pl_module.data_indices.model.output.name_to_index[name]: name for name in self.config.diagnostics.plot.parameters
        }
        if self.spread_skill.num_updates != 0:
            rmse, spread, bins_rmse, bins_spread = (r for r in self.spread_skill.compute())
            rmse, spread, bins_rmse, bins_spread = (
                safe_cast_to_numpy(rmse),
                safe_cast_to_numpy(spread),
                safe_cast_to_numpy(bins_rmse),
                safe_cast_to_numpy(bins_spread),
            )
            fig = plot_spread_skill(plot_parameters_dict, (rmse, spread), self.spread_skill.time_step)
            self._output_figure(
                trainer,
                fig,
                epoch=trainer.current_epoch,
                tag="ens_spread_skill",
                exp_log_tag=f"val_spread_skill_{pl_module.global_rank}",
            )
            fig = plot_spread_skill_bins(plot_parameters_dict, (bins_rmse, bins_spread), self.spread_skill.time_step)

            if self.subplots_kwargs is not None:
                fig.subplots_adjust(**self.subplots_kwargs)
            else:
                fig.tight_layout()
                self.subplots_kwargs = {
                    par: getattr(fig.subplotpars, par) for par in ["left", "right", "bottom", "top", "wspace", "hspace"]
                }

            self._output_figure(
                trainer,
                fig,
                epoch=trainer.current_epoch,
                tag="ens_spread_skill_bins",
                exp_log_tag=f"val_spread_skill_bins_{pl_module.global_rank}",
            )


