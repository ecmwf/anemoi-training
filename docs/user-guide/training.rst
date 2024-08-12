##########
 Training
##########

.. _restart target:

Restarting a model train
------------------------

Whether it's because the training has exceeded the time limit on an HPC system or because the user wants to fine-tune the model from a specific point in the training, it may be necessary at certain points to restart the model training.

This can be done by setting `config.training.run_id` in the config file to be the *run_id* of the run that is being restarted. In this case the new checkpoints will go in the same folder as the old checkpoints. 
If the user does not want this then they can instead set `config.training.fork_run_id` in the config file to the *run_id* of the run that is being restarted. In this case the old run will be unaffected and the new checkpoints will go in to a new folder with a new run_id. The user might want to do this if they want to start multiple new runs from 1 old run.

The above will restart the model training from where the old run finished training. However if the user wants to restart the model from a specific point they can do this by setting `config.hardware.files.warm_start` to be the checkpoint they want to restart from..

Learning rate
-------------

Anemoi training uses the `CosineLRScheduler` from pytorch as it's learning rate scheduler. The user can configure the maximum learning rate by setting `config.training.lr.rate`. Note that this learning rate is configured by the number of GPUs where  `global_learning_rate = config.training.lr.rate * num_gpus_per_node * num_nodes / gpus_per_model`.

The user can also control the rate at which the learning rate decreases by setting the total number of iterations through `config.training.lr.iterations` and the minimum learning rate reached through `config.training.lr.min`. Note that the minimum learning rate is not scaled by the number of GPUs.

Loss function scaling
---------------------

It is possible to change the weighting given to each of the variables in the loss function by changing `config.training.loss_scaling.pl.<pressure level variable>` and `config.training.loss_scaling.sfc.<surface variable>`. 

It is also possible to change the scaling given to the pressure levels using `config.training.pressure_level_scaler`. For almost all applications, upper atmosphere pressure levels should be given lower weighting than the lower atmosphere pressure levels (i.e. pressure levels nearer to the surface). 
By default anemoi-training uses a ReLU Pressure Level scaler with a minimum weighting of 0.2 (i.e. no pressure level has a weighting less than 0.2).

Rollout
-------

In the first stage of training, standard practice is to train the model on a 6 hour interval. Once this is completed, in the second stage of training, it is advisable to *rollout* and fine-tune the model error at longer leadtimes too.
Generally for medium range forecasts, rollout is performed on 12 forecast steps (equivalent to 72 hours) incrementally. In other words, at each epoch another forecast step is added to the error term.

Rollout requires the model training to be restarted so the user should make sure to set `config.training.run_id` equal to the run-id of the first stage of training.

Note, in the standard set-up, rollout is performed at the minimum learning rate and the number of batches used is reduced (using `config.dataloader.training.limit_batches`) to prevent any overfit to specific timesteps.

To start rollout set `config.training.rollout.epoch_increment` equal to 1 (thus increasing the rollout step by 1 at every epoch) and set a maximum rollout by setting `config.training.rollout.max` (usually set to 12).
