#################
 Training your first model
#################

Once anemoi-training is installed, you can run your first model with

.. code:: bash

  aifs-train

which will use the default model configurations.

For the ECMWF, Leonardo and Meluxina HPC Systems we have default configurations set up which can be specified as follows.

.. code:: bash

    aifs-train hardware=[atos|leo|mlux]-slurm