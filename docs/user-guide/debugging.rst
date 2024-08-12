#################
 Troubleshooting
#################

To troubleshoot errors when trying to train a model for the first time, it is advisable to the debug configuration `anemoi/training/config/debug.yaml`. This runs a small model and trains on only a limited number of batches per epoch and so should mean that your error message appears more quickly.

If you are using your own configuration, when troubleshooting you should limit the number of batches (using `dataloader.limit_batches.training = 100` and `dataloader.limit_batches.validation=100`) and run on 1 gpu to check if the error is coming from the parallelisation of the data/model.

It may also be a good idea to make the callbacks asynchronous (by setting `config.diagnostics.plot.asynchronous` to False), as this makes the error messages generally easier to understand. Furthermore the plotting callbacks can be switched off by setting `config.diagnostics.plot.enabled` to *False* 
or all the callbacks can be completely switched off by setting `config.diagnostics.eval.enabled` to *False*. This can help with isolating the source of the issue.