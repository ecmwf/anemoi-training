# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import csv
import logging
import os
import re
from pathlib import Path
from typing import Any
from typing import Optional

import memray
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import wandb
from memray import FileFormat
from memray import FileReader
from memray.reporters.table import TableReporter
from omegaconf import DictConfig
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT

import anemoi.training

LOGGER = logging.getLogger(__name__)

PROFILER_ACTIONS = [
    r"\[Strategy]\w+\.batch_to_device",
    r"\[Strategy]\w+\.backward",
    r"\[Strategy]\w+\.validation_step",
    r"\[Strategy]\w+\.batch_to_device",
    "run_training_epoch",
    "run_training_batch",
    r"\[_EvaluationLoop\]\.\w+",
    r"\[_TrainingEpochLoop\]\.\w+",
    r"\[LightningDataModule]\w+\.train_dataloader",
    r"\[LightningDataModule]\w+\.val_dataloader",
    r"\[LightningDataModule]\w+\.state_dict",
    r"\[LightningDataModule]\w+\.setup",
    r"\[LightningDataModule]\w+\.prepare_data",
    r"\[LightningDataModule]\w+\.teardown",
    r"\[LightningModule]\w+\.optimizer_step",
    r"\[LightningModule]\w+\.configure_gradient_clipping",
    r"\[LightningModule]\w+\.on_validation_model_eval",
    r"\[LightningModule]\w+\.optimizer_zero_grad",
    r"\[LightningModule]\w+\.transfer_batch_to_device",
    r"\[LightningModule]\w+\.on_validation_model_train",
    r"\[LightningModule]\w+\.configure_optimizers",
    r"\[LightningModule]\w+\.lr_scheduler_step",
    r"\[LightningModule]\w+\.configure_sharded_model",
    r"\[LightningModule]\w+\.setup",
    r"\[LightningModule]\w+\.prepare_data",
    r"\[Callback\](.*Plot*)",
    r"\[Callback\](.*Checkpoint*)",
]

GPU_METRICS_DICT = {
    "GPU device utilization (%)": "gpu",
    "GPU memory use (%)": "memory",
    "GPU memory allocated (%)": "memoryAllocated",
    "GPU memory allocated (GB)": "memoryAllocatedBytes",
}


def get_wandb_metrics(run_id_path: str) -> (pd.DataFrame, dict):
    """Fetches system metrics and metadata from a W&B run."""
    run = wandb.Api().run(run_id_path)
    system_metrics = run.history(stream="events")
    metadata_dict = run.metadata
    system_metrics = system_metrics.dropna()
    return system_metrics, metadata_dict


def summarize_gpu_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Given the System Metrics DataFrame, summarized the GPU metrics.

    - gpu.{gpu_index}.memory - GPU memory utilization in percent for each GPU
    - gpu.{gpu_index}.memoryAllocated - GPU memory allocated as a percentage of the total available memory for each GPU
    - gpu.{gpu_index}.memoryAllocatedBytes - GPU memory allocated in bytes for each GPU
    - gpu.{gpu_index}.gpu - GPU utilization in percent for each GPU
    """
    average_metric = {}
    col_names = df.columns
    for gpu_metric_name, gpu_metric in GPU_METRICS_DICT.items():
        pattern = rf"system.gpu.\d.{gpu_metric}$"
        sub_gpu_cols = [string for string in col_names if re.match(pattern, string)]
        metrics_per_gpu = df[sub_gpu_cols].mean(axis=0)
        if gpu_metric == "memoryAllocatedBytes":
            metrics_per_gpu = metrics_per_gpu * 1e-9
        average_metric[gpu_metric_name] = metrics_per_gpu.mean()
        # Just add metrics per gpu to the report if we have more than 1 GPU
        if metrics_per_gpu.shape[0] > 1:
            metrics_per_gpu.index = ["   " + index for index in metrics_per_gpu.index]
            average_metric.update(dict(metrics_per_gpu))
    return average_metric


def summarize_wandb_system_metrics(run_id_path: str) -> dict[str, float]:
    r"""Summarizes the System metrics from a W&B run.

    Some of the metrics included are:
      - cpu.{}.cpu_percent - CPU usage of the system on a per-core basis.
      - system.memory - Represents the total system memory usage as a percentage of the total available memory.
      - system.cpu - Percentage of CPU usage by the process, normalized by the number of available CPUs
      - system.disk.\\.usageGB - (Represents the total system disk usage in gigabytes (GB))
      - system.proc.memory.percent - Indicates the memory usage of the process as a percentage of the total available memory

    More information about W&B system metrics can be found here:
    https://docs.wandb.ai/guides/app/features/system-metrics
    """
    system_metrics_df, metadata_dict = get_wandb_metrics(run_id_path)

    col_names = system_metrics_df.columns
    system_metrics = {}

    n_cpus = metadata_dict["cpu_count"]
    cpu_cols = list(filter(lambda k: "cpu." in k, col_names))
    system_metrics["avg CPU usage (%)"] = (system_metrics_df[cpu_cols].sum(axis=1) / n_cpus).mean()

    system_metrics_gpu = summarize_gpu_metrics(system_metrics_df)
    system_metrics.update(system_metrics_gpu)

    system_metrics["avg Memory usage (%)"] = system_metrics_df["system.memory"].mean()
    system_metrics["avg Disk usage (GB)"] = system_metrics_df["system.disk.\\.usageGB"].mean()
    system_metrics["avg Disk usage  (%)"] = system_metrics_df["system.disk.\\.usagePercent"].mean()

    system_metrics["execution time (sec)"] = system_metrics_df["_runtime"].iloc[-1]  # in seconds
    return system_metrics


class BenchmarkProfiler(Profiler):
    """Custom PyTorch Lightning profiler for benchmarking.

    Parameters
    ----------
    config : DictConfig
        Configuration object.

    Attributes
    ----------
    dirpath : Path
        Path to the profiler directory.
    benchmark_filename : Path
        Path to the benchmark profiler file.
    time_profiler : SimpleProfiler
        Simple profiler for time measurements.
    pid : int
        Process ID.
    memfile_name : Path
        Path to the memory profiler file.
    memory_profiler : memray.Tracker
        Memory profiler.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        self.config = config
        self.dirpath = Path(self.config.hardware.paths.profiler)
        self.dirpath.mkdir(parents=True, exist_ok=True)

        self.benchmark_filename = Path(self.dirpath, "aifs-benchmark-profiler.csv")

        self._create_profilers()

    @rank_zero_only
    def _create_output_file(self) -> None:
        """Creates the output file to aggregate the memory profiling results."""
        fields = ["category", "size (MiB)", "function", "group", "pid"]
        with self.benchmark_filename.open("w") as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def _create_profilers(self) -> None:
        """Creates profilers for time and memory measurements."""
        self.time_profiler = SimpleProfiler(
            dirpath=self.dirpath,
        )
        self.pid = os.getpid()

        self.memfile_name = Path(self.dirpath, f"aifs-benchmark-mem-profiler_{self.pid}.bin")
        self.memory_profiler = memray.Tracker(self.memfile_name, file_format=FileFormat.AGGREGATED_ALLOCATIONS)
        self._create_output_file()

    def start(self, action_name: str) -> None:
        """Starts recording time for a specific action.

        Parameters
        ----------
        action_name : str
            Name of the action.
        """
        self.time_profiler.start(action_name)

    def stop(self, action_name: str) -> None:
        """Stops recording time for a specific action.

        Parameters
        ----------
        action_name : str
            Name of the action.
        """
        self.time_profiler.stop(action_name)

    def _trim_time_report(self, recorded_actions: dict) -> dict[str, float]:
        all_actions_names = recorded_actions.keys()
        df = pd.DataFrame({"Strings": all_actions_names})
        combined_pattern = "|".join(PROFILER_ACTIONS)
        filtered_df = df[df["Strings"].str.contains(combined_pattern, regex=True, na=False)]
        trimmed_actions_names = filtered_df["Strings"].tolist()
        cleaned_recorded_actions = {key: recorded_actions[key] for key in trimmed_actions_names}
        return cleaned_recorded_actions

    def get_time_profiler_df(self, precision: int = 5) -> pd.DataFrame:
        """Retrieves a DataFrame with time profiling information.

        Parameters
        ----------
        precision : int
            Precision for rounding, by default 5

        Returns
        -------
        pd.DataFrame
            DataFrame with time profiling information.
        """
        self.time_profiler.recorded_durations = self._trim_time_report(
            recorded_actions=self.time_profiler.recorded_durations
        )
        time_df = pd.DataFrame(self.time_profiler.recorded_durations.items())
        time_df[2] = time_df[1].apply(len)
        time_df[3] = time_df[1].apply(np.mean)
        time_df[1] = time_df[1].apply(sum)
        time_df.columns = ["name", "total_time", "n_calls", "avg_time"]

        def replace_function(value):
            # Replace 'apple' with 'fruit'
            value = re.sub(r"\{.*?\}", "", value)  # Remove anything between brackets
            return value

        time_df.to_csv(Path(self.config.hardware.paths.profiler, "time_profiler_no_replace.csv"))
        time_df["name"] = time_df["name"].apply(replace_function)
        pattern = r"\[(.*?)\]|(.*)"
        time_df["category"] = time_df["name"].str.extract(pattern, expand=False)[0].fillna(time_df["name"])

        pattern = re.compile(r"\[Callback\](.*?)\.")
        # Apply the regular expression to the column
        callbacks_subcategories = "*Callback_" + time_df[time_df["category"] == "Callback"]["name"].str.extract(pattern)
        indexer = time_df[time_df["category"] == "Callback"].index
        time_df.loc[indexer, "category"] = callbacks_subcategories[0].tolist()
        time_df.to_csv(Path(self.config.hardware.paths.profiler, "time_profiler_complete.csv"))

        # Check if 'Callback' is present in the 'category' column
        time_df["is_callback"] = time_df["category"].str.contains("Callback", case=False)

        # Group by the 'is_callback' column and apply groupby operation only on rows with 'Callback' in 'category'
        grouped_data = (
            time_df[time_df["is_callback"]]
            .groupby("category")
            .agg({"n_calls": np.sum, "avg_time": np.sum, "total_time": np.sum})
            .reset_index()
        )
        grouped_data["name"] = grouped_data["category"]

        time_df = pd.concat([time_df[~time_df["is_callback"]], grouped_data])
        time_df = time_df.drop("is_callback", axis=1)
        time_df = time_df.round(precision)
        time_df = time_df.sort_values(by="category", ascending=False)
        return time_df

    def _generate_memray_df(self) -> pd.DataFrame:
        """Generates dataframe from memray tracking results.

        For each node/process we convert the tracking results to a dataframe just
        keeping the  high watermark allocations.

        Returns
        -------
        pd.DataFrame
            Memory profiler data.
        """
        self.memory_profiler.__exit__(None, None, None)
        memfile_tracking = FileReader(self.memfile_name)
        memory_allocations = list(memfile_tracking.get_high_watermark_allocation_records())
        table = TableReporter.from_snapshot(memory_allocations, memory_records=[], native_traces=False)
        df = pd.DataFrame(table.data)
        memfile_tracking.close()
        return df

    def _aggregate_per_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates memory profiling information per category.

        Each stack_trace tracked by memray is separated into parts
        - first part points to the path of the library - referred as category
        - second part is the exact name of the function of this script

        Since we can have traces coming from the same script but referring
        to multiple functions in that script, we aggregate those and in the
        'function' entry we just keep the function that has a higher memory
        consumption.
        """
        pattern = r"^(.*?) at (.*?)\.py"
        new_cols = df.loc[:, "stack_trace"].str.extract(pattern)
        df = df.assign(function=new_cols[0], category=new_cols[1])
        df = df.drop("stack_trace", axis=1)
        df_agg = df.groupby("category").apply(
            lambda x: pd.Series(
                {
                    "size (MiB)": x["size (MiB)"].sum(),
                    "function": x.loc[x["size (MiB)"].idxmax()]["function"],
                },
            ),
        )
        df_agg.reset_index(inplace=True)
        return df_agg

    def _trim_memray_df(self, memray_df: pd.DataFrame, precision: int = 5, n_items: int = 10) -> pd.DataFrame:
        """Trims and processes the memray DataFrame.

        Necessary since Memray tracks memory allocations across different files.

        all the script, we group those allocations in two categories:
            - aifs-operations: coming from functions included in this repository
            - general-operations: coming from functions from other python libraries

        Parameters
        ----------
        memray_df : pd.DataFrame
            Input DataFrame from memray.
        precision : int, optional
            Precision for rounding, by default 5
        n_items : int, optional
            Number of top memory-consuming items to include, by default 10

        Returns
        -------
        pd.DataFrame
            Compiled dataframe from memray data.
        """
        cleaned_memray_df = memray_df.drop("tid", axis=1)
        cleaned_memray_df = cleaned_memray_df.drop("allocator", axis=1)

        # For readibility, we cut the paths to just display the relevant package info
        module_path = anemoi.training.__path__[0].replace("aifs-mono/aifs", "")
        env_path = pl.__path__[0].replace("pytorch_lightning", "")
        base_env_path = pl.__path__[0].replace("/site-packages/pytorch_lightning", "")

        cleaned_memray_df["stack_trace"] = cleaned_memray_df["stack_trace"].apply(lambda x: x.replace(module_path, ""))
        cleaned_memray_df["stack_trace"] = cleaned_memray_df["stack_trace"].apply(lambda x: x.replace(env_path, ""))
        cleaned_memray_df["stack_trace"] = cleaned_memray_df["stack_trace"].apply(
            lambda x: x.replace(base_env_path, "")
        )

        cleaned_memray_df["size (MiB)"] = cleaned_memray_df["size"] * 9.5367e-7
        cleaned_memray_df.sort_values("size (MiB)", ascending=False, inplace=True)
        cleaned_memray_df = cleaned_memray_df.drop("size", axis=1)

        top_most_memory_consuming_df = cleaned_memray_df[~cleaned_memray_df["stack_trace"].str.contains("aifs")].head(
            n_items
        )
        top_most_memory_consuming_df = self._aggregate_per_category(top_most_memory_consuming_df)

        aifs_memray = cleaned_memray_df[cleaned_memray_df["stack_trace"].str.contains("aifs")]
        aifs_memray = self._aggregate_per_category(aifs_memray)

        aifs_memray["group"] = "aifs-operations"
        top_most_memory_consuming_df["group"] = "general-operations"

        merged_memory_df = pd.concat([top_most_memory_consuming_df, aifs_memray])
        merged_memory_df = merged_memory_df.round(precision)
        return merged_memory_df

    def teardown(self, stage: Optional[str]) -> None:
        """Clean up before closing the profiler.

        Before closing the profiler, performs the cleanup operations and writes memray
        data to the common benchmark file.
        """
        memray_df = self._generate_memray_df()
        cleaned_memray_df = self._trim_memray_df(memray_df)

        cleaned_memray_df["pid"] = self.pid
        cleaned_memray_df.to_csv(self.benchmark_filename, mode="a", index=False, header=False)

    @rank_zero_only
    def get_memory_profiler_df(self) -> pd.DataFrame:
        """Retrieves the memory profiler data as a DataFrame.

        Aggregates the results coming from multiple nodes/processes.

        Returns
        -------
        pd.DataFrame
            Memory profiler data.
        """
        mem_df = pd.read_csv(self.benchmark_filename)
        return (
            mem_df.groupby(["category", "group", "function"])
            .apply(
                lambda x: pd.Series(
                    {
                        "size (MiB)": x["size (MiB)"].mean(),
                        "pid": len(set(x["pid"])),
                    },
                ),
            )
            .reset_index()
            .sort_values("size (MiB)", ascending=False)
        )

    def __del__(self) -> None:
        self.teardown(stage=self._stage)
        self.memfile_name.unlink()


class ProfilerProgressBar(TQDMProgressBar):
    """Custom PyTorch Lightning progress bar with profiling functionality.

    Attributes
    ----------
    validation_rates : list[float]
        List to store validation rates (it/s).
    training_rates : list[float]
        List to store training rates (it/s).
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.validation_rates = []
        self.training_rates = []

    def _extract_rate(self, pbar) -> float:
        """Extracts the iteration rate from the progress bar.

        Parameters
        ----------
        pbar : tqdm
            The progress bar.

        Returns
        -------
        float
            The iteration rate.
        """
        return (pbar.format_dict["n"] - pbar.format_dict["initial"]) / pbar.format_dict["elapsed"]

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Appends the rate from the progress bar to the list of 'training_rates'."""
        batch_idx + 1
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.train_progress_bar.format_dict["n"] != 0:
            self.training_rates.append(self._extract_rate(self.train_progress_bar))

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Append rate from the progress bar to the list of 'validation_rates'."""
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.val_progress_bar.format_dict["n"] != 0:
            self.validation_rates.append(self._extract_rate(self.val_progress_bar))

    @rank_zero_only
    def summarize_metrics(self, config) -> dict[str, float]:
        """Summarizes and returns speed metrics based on training and validation rates.

        Parameters
        ----------
        config : Config
            The configuration object.

        Returns
        -------
        dict
            A dictionary containing speed metrics.
        """
        speed_metrics = {}

        batch_size_tr = config.dataloader.batch_size.training
        batch_size_val = config.dataloader.batch_size.validation

        training_rates_array = np.array(self.training_rates)
        speed_metrics["training_avg_throughput"] = training_rates_array.mean()
        speed_metrics["training_avg_throughput_per_sample"] = training_rates_array.mean() / batch_size_tr

        validation_rates_array = np.array(self.validation_rates)
        speed_metrics["validation_avg_throughput"] = validation_rates_array.mean()
        speed_metrics["validation_avg_throughput_per_sample"] = validation_rates_array.mean() / batch_size_val

        return speed_metrics
