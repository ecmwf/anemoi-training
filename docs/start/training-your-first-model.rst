###########################
 Training your first model
###########################

Once Anemoi training is installed, you can run your first model with

.. code:: bash

   anemoi-training train

which will use the default model configurations with missing values.

The training script will intentionally crash as it does not know where
your data is stored.

These missing values in the configuration are placeholders for the user
to fill in marked with `???`. You can find the default configurations in
the `anemoi-training` repository under `src/anemoi/training/configs/`.
Alternatively, the next section will show you how to generate a user
config file.
