# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import pytorch_lightning as pl

LOGGER = logging.getLogger(__name__)


class CheckVariableOrder(pl.callbacks.Callback):
    """Check the order of the variables in a pre-trained / fine-tuning model."""

    def on_epoch_start(self, trainer: pl.Trainer, *args, **kwargs) -> None:
        del args, kwargs
        LOGGER.info("Checking the order of the variables in the model.")
        model_name_to_index = trainer.data_indices.name_to_index
        data_name_to_index = trainer.datamodule.data_indices

        if model_name_to_index != data_name_to_index:
            keys1 = set(model_name_to_index.keys())
            keys2 = set(data_name_to_index.keys())

            # Find keys unique to each dictionary
            only_in_model = {key: model_name_to_index[key] for key in (keys1 - keys2)}
            only_in_data = {key: data_name_to_index[key] for key in (keys2 - keys1)}

            # Find common keys
            common_keys = keys1 & keys2

            # Compare values for common keys
            different_values = {
                k: (model_name_to_index[k], data_name_to_index[k])
                for k in common_keys
                if model_name_to_index[k] != data_name_to_index[k]
            }

            LOGGER.warning(
                "The variables in the model do not match the variables in the data. "
                "If you're fine-tuning or pre-training, you may have to adjust the variable order in your config.",
            )
            if only_in_model:
                LOGGER.warning("Variables only in model: %s", only_in_model)
            if only_in_data:
                LOGGER.warning("Variables only in data: %s", only_in_data)
            if different_values:
                LOGGER.error("Detected a different sort order of the same variables: %s.", different_values)
                LOGGER.error(
                    "Please adjust the variable order in your config, you may need to use the "
                    "'reorder' key in the dataloader config. With:\n `reorder: %s`",
                    model_name_to_index,
                )
