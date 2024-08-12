#####################
 Basic Configuration
#####################

Anemoi-training is configurable using the hydra-config system.

The configuration options for the model are located `/src/anemoi/training/config/`. 
They are split across files based on topic. Most of the time, the default configs can be used, with specific ones being overridden. 

In order to use your model config file `--config-name=` should be added to the training command like so:

.. code:: bash

    aifs-train --config-name=...

A typical config file will start with specifying the default config settings at the top as follows:

.. code:: bash

    # the default config settings are given below
    defaults:
    - hardware: atos
    - data: zarr
    - dataloader: default
    - model: transformer
    - graph: default
    - training: default
    - diagnostics: eval_rollout
    - override hydra/job_logging: none
    - override hydra/hydra_logging: none
    - _self_

These are group configs for each section. The options after the defaults are then used to override the configs, by assigning new features and keywords. 

For example to change from default GPU count:

.. code:: bash

    hardware:
        num_gpus_per_node: 1

Key config options which must be overridden by new users are:

- `hardware.num_gpus_per_model`: This specifies model paralellism. When running large models on many nodes, consider increasing this. Clusters might have a different value.
- `hardware.paths.data`: Location of base directory where datasets are stored
- `hardware.paths.output`: Location of output directory
- `hardware.files`: Name of datasets used for training.
- `hardware.files.graph`: If you have pre-computed a specific graph, specify this here. Otherwise, a new graph will be constructed on the fly.

If you would like to log your model run on ML-flow to monitor the progress of training, you should set:

.. code:: bash

    diagnostics:
        log:
            mlflow:
                enabled: True
                experiment_name: *aifs*
                run_name: *my_first_run*


The value you set for experiment_name is to create a group for all your runs. run_name should be something uniquely describing the specific training run you are doing.

Command-line config overrides
-----------------------------

It is also possible to use command line config overrides. We can switch out group configs using 

.. code:: bash

    aifs-train hardware=atos_slurm

or override individual config entries such as

.. code:: bash

    aifs-train diagnostics.log.mlflow.enabled=False

or combine everything together

.. code:: bash

    aifs-train --config-name=<user-defined-config> hardware=atos_slurm diagnostics.log.mlflow.enabled=False
