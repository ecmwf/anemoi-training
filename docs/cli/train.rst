train
======

Use this command to create a train a model:

.. code-block:: bash

    % anemoi-training train config.yaml

The command will read the default configuration and override it with the values in the provided configuration file.
The configuration file should be a YAML file with the structure defined in the `Configuration` section. The file `config.yaml` will typically destribes
the model to be trained, the dataset to be used, and the training hyperparameters:

.. literalinclude:: train.yaml
    :language: yaml

You can  provide more that one configuration file, in which case the values will be merged in the order they are provided. A typical usage would be
to split the training configurations into model description, training hyperparameters and runtime options

.. code-block:: bash

    % anemoi-training train model.yaml hyperparameters.yaml slurm.yaml

Furthermore, you can also provide values directly on the command line, which will override any values in the configuration files:

.. code-block:: bash

    % anemoi-training train config.yaml tracker.mlflow.tracking_uri=http://localhost:5000

If the file `~/.config/anemoi/train.yaml` exists, it will be loaded after the defaults and before any other configuration file.
This allows you to provide values such as passwords or other sensitive information that you do not want to store a git repository.

*********************
 Command line usage
*********************

.. argparse::
    :module: anemoi.training.__main__
    :func: create_parser
    :prog: anemoi-training
    :path: train
