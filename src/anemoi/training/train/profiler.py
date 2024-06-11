import logging
import os
import warnings
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only
from rich.console import Console

from anemoi.training.diagnostics.profilers import BenchmarkProfiler
from anemoi.training.diagnostics.profilers import ProfilerProgressBar
from anemoi.training.train.train import AIFSTrainer

LOGGER = logging.get_code_logger(__name__)

console = Console(record=True, width=200)


class AIFSProfiler(AIFSTrainer):
    """Profiling for AIFS."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        if (not self.config.diagnostics.log.wandb.enabled) or (self.config.diagnostics.log.wandb.offline):
            warnings.warn("Warning: W&B logging is deactivated so no system report would be provided")

    @staticmethod
    def print_report(title: str, dataframe: pd.DataFrame, color="white", emoji="") -> None:
        console.print(f"[bold {color}]{title}[/bold {color}]", f":{emoji}:")
        console.print(dataframe.to_markdown(headers="keys", tablefmt="psql"), end="\n\n")

    @staticmethod
    def print_title() -> None:
        console.print("[bold magenta] Benchmark Profiler Summary [/bold magenta]!", ":book:")

    @staticmethod
    def print_metadata() -> None:
        console.print(f"[bold blue] SLURM NODE(s) {os.environ['HOST']} [/bold blue]!")
        console.print(f"[bold blue] SLURM JOB ID {os.environ['SLURM_JOB_ID']} [/bold blue]!")
        console.print(
            f"[bold blue] TIMESTAMP {datetime.now(datetime.timezone.utc).strftime('%d/%m/%Y %H:%M:%S')} [/bold blue]!"
        )

    @rank_zero_only
    def print_benchmark_profiler_report(
        self,
        speed_metrics_df: Optional[pd.DataFrame],
        time_metrics_df: Optional[pd.DataFrame],
        memory_metrics_df: Optional[pd.DataFrame],
        system_metrics_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.print_title()
        self.print_metadata()

        warnings.warn(
            "INFO: Time Report metrics represent single-node metrics (rank-0 process) (not multi-node aggregated metrics)",
        )
        warnings.warn("INFO: Metrics with a * symbol, represent the value after aggregating all steps")
        self.print_report("Time Profiling", time_metrics_df, color="green", emoji="alarm_clock")

        warnings.warn(
            "INFO: Speed Report metrics represent single-node metrics (rank-0 process) (not multi-node aggregated metrics)",
        )
        self.print_report("Speed Profiling", speed_metrics_df, color="yellow", emoji="racing_car")

        if memory_metrics_df is not None:
            console.print(memory_metrics_df)
            warnings.warn("INFO: Memory Report metrics represent metrics aggregated across all nodes")
            self.print_report("Memory Profiling", memory_metrics_df, color="purple", emoji="floppy_disk")

        if system_metrics_df is not None:
            self.print_report("System Profiling", system_metrics_df, color="purple", emoji="desktop_computer")

    @staticmethod
    def write_benchmark_profiler_report() -> None:
        console.save_html("report.html")

    @staticmethod
    def to_df(sample_dict: dict[str, float], precision: str = ".5") -> pd.DataFrame:
        df = pd.DataFrame(sample_dict.items())
        df.columns = ["metric", "value"]
        df.value = df.value.apply(lambda x: f"%{precision}f" % x)
        return df

    @cached_property
    def speed_profile(self):
        """Speed profiler Report.

        Get speed metrics from Progress Bar for training and validation.
        """
        if self.config.diagnostics.benchmark_profiler.speed:
            # Find the first ProfilerProgressBar callback.
            for callback in self.callbacks:
                if isinstance(callback, ProfilerProgressBar):
                    speed_metrics_dict = callback.summarize_metrics(self.config)
                    break
            else:
                raise ValueError("No ProfilerProgressBar callback found.")

            # Calculate per_sample metrics
            speed_metrics_dict["avg_training_dataloader_throughput"] = (
                1
                / np.array(
                    self.profiler.time_profiler.recorded_durations["[_TrainingEpochLoop].train_dataloader_next"]
                ).mean()
            )
            speed_metrics_dict["avg_training_dataloader_throughput_per_sample"] = (
                speed_metrics_dict["avg_training_dataloader_throughput"] / self.config.dataloader.batch_size.training
            )

            speed_metrics_dict["avg_validation_dataloader_throughput"] = (
                1 / np.array(self.profiler.time_profiler.recorded_durations["[_EvaluationLoop].val_next"]).mean()
            )
            speed_metrics_dict["avg_validation_dataloader_throughput_per_sample"] = (
                speed_metrics_dict["avg_validation_dataloader_throughput"]
                / self.config.dataloader.batch_size.validation
            )

            return self.to_df(speed_metrics_dict)
        return None

    def _get_logger(self):
        if (self.config.diagnostics.log.wandb.enabled) and (not self.config.diagnostics.log.wandb.offline):
            logger_name = "wandb"
            logger = self.wandb_logger
        elif self.config.diagnostics.log.tensorboard.enabled:
            logger_name = "tensorboard"
            logger = self.tensorboard_logger
        elif self.config.diagnostics.log.mlflow.enabled:
            logger_name = "mlflow"
            logger = self.mlflow_logger
        else:
            LOGGER.warning("No logger enabled for system profiler")
            return None
        return logger_name, logger

    @cached_property
    def system_profile(self):
        """System Profiler Report."""
        if self.config.diagnostics.benchmark_profiler.system:
            logger_name, logger = self._get_logger()
            return self.profiler.get_system_profiler_df(logger_name=logger_name, logger=logger)
        return None

    @cached_property
    def memory_profile(self):
        """Memory Profiler Report."""
        if self.config.diagnostics.benchmark_profiler.memory:
            return self.profiler.get_memory_profiler_df()
        return None

    @cached_property
    def time_profile(self):
        """Time Profiler Report."""
        if self.config.diagnostics.benchmark_profiler.time:
            return self.profiler.get_time_profiler_df()
        return None

    @rank_zero_only
    def export_to_logger(self) -> None:
        if (self.config.diagnostics.log.wandb.enabled) and (not self.config.diagnostics.log.wandb.offline):
            self.to_wandb()

        elif self.config.diagnostics.log.mlflow.enabled:
            self.to_mlflow()

    @rank_zero_only
    def report(self) -> str:
        """Print report to console."""
        if (not self.config.diagnostics.log.wandb.enabled) or (self.config.diagnostics.log.wandb.offline):
            self.print_benchmark_profiler_report(
                speed_metrics_df=self.speed_profile,
                time_metrics_df=self.time_profile,
                memory_metrics_df=self.memory_profile,
            )
        else:
            self._close_logger()
            warnings.warn(
                "INFO: System report is provided from W&B system metrics which are single-node metrics (rank-0 process)"
            )

            self.print_benchmark_profiler_report(
                speed_metrics_df=self.speed_profile,
                time_metrics_df=self.time_profile,
                memory_metrics_df=self.memory_profile,
                wandb_memory_metrics_df=self.wandb_profile,
            )

    @rank_zero_only
    def to_wandb(self) -> None:
        """Log report into W&B."""
        LOGGER.info("logging to W&B Profiler report")
        self.write_benchmark_profiler_report()
        import wandb
        from pytorch_lightning.loggers.wandb import WandbLogger

        logger = WandbLogger(
            project=self.run_dict["project"],
            entity=self.run_dict["entity"],
            id=self.run_dict["id"],
            offline=self.config.diagnostics.log.wandb.offline,
            resume=self.run_dict["id"],
        )

        logger.experiment.log({"speed_metrics_report": wandb.Table(dataframe=self.speed_profile)})
        logger.experiment.log({"wandb_memory_metrics_report": wandb.Table(dataframe=self.wandb_profile)})
        logger.experiment.log({"time_metrics_report": wandb.Table(dataframe=self.time_profile)})
        logger.experiment.log({"memory_metrics_report": wandb.Table(dataframe=self.memory_profile)})
        with Path("report.html").open() as f:
            logger.experiment.log({"reports_benchmark_profiler": wandb.Html(f)})
        logger.experiment.finish()

    @cached_property
    def callbacks(self) -> list[pl.callbacks.Callback]:
        callbacks = super().callbacks
        callbacks.append(ProfilerProgressBar())
        # from pytorch_lightning.callbacks import RichModelSummary
        # from pytorch_lightning.callbacks import DeviceStatsMonitor

        # device_stats = DeviceStatsMonitor()
        # callbacks.append(RichModelSummary())
        # callbacks.append(device_stats)
        return callbacks

    @cached_property
    def profiler(self) -> BenchmarkProfiler:
        return BenchmarkProfiler(self.config)

    def update_paths(self) -> None:
        """Update the paths in the configuration."""
        super().update_paths()
        self.config.hardware.paths.profiler = Path(self.config.hardware.paths.profiler, self.run_id)
        LOGGER.info(self.config.hardware.paths.profiler)
        LOGGER.info(self.config.hardware.paths.plots)
        LOGGER.info(self.config.hardware.paths.checkpoints)

    def _close_logger(self) -> None:
        if (self.config.diagnostics.log.wandb.enabled) and (not self.config.diagnostics.log.wandb.offline):
            # We need to close the W&B logger to be able to read the System Metrics
            self.wandb_logger.experiment.finish()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    trainer_aifs = AIFSProfiler(config)

    trainer_aifs.train()

    trainer_aifs.report()

    trainer_aifs.export_to_logger()
