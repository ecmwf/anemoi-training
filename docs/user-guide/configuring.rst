##########################
 Configuring the Training
##########################

Anemoi training is set up in a way that you should be able to modify key
components of both the models and training without changing the code.

A basic introduction to the configuration system is provided in the
`getting started <start/hydra-intro>`_ section. This section will go
into more detail on how to configure the training pipeline.

***********************
 Default Config Groups
***********************

A typical config file will start with specifying the default config
settings at the top as follows:

.. code:: yaml

   defaults:
   - data: zarr
   - dataloader: native_grid
   - diagnostics: eval_rollout
   - hardware: example
   - graph: multi_scale
   - model: gnn
   - training: default
   - _self_

These are group configs for each section. The options after the defaults
are then used to override the configs, by assigning new features and
keywords.

For example to change from default GPU count:

.. code:: yaml

   hardware:
       num_gpus_per_node: 1

*******************************
 Command-line config overrides
*******************************

It is also possible to use command line config overrides. We can switch
out group configs using

.. code:: bash

   anemoi-training train model=transformer

or override individual config entries such as

.. code:: bash

   anemoi-training train diagnostics.plot.enabled=False

or combine everything together

.. code:: bash

   anemoi-training train --config-name=debug.yaml model=transformer diagnostics.plot.enabled=False
