###############
 Configuration
###############

The hydra config package not only allows the user to define configurable
options but also to instantiate objects from the config. This allows for
greater flexibility within the code.

The code using the instantiated object only knows the interface which
remains constant, but the behavior is determined by the actual object
instance. Hydra provides ``hydra.utils.instantiate()`` for instantiating
objects and ``hydra.utils.call()`` for calling functions.

anemoi-training makes use of this functionality to define, for example,
new loss scalings (see ``config.training.pressure_level_scaler``).

Below is a simple example to show how users can instantiate objects from
the config:

**Example**

Suppose we have the following class:

.. code:: python

   class Optimiser:
       algorithm: str
       learning_rate: float

   def __init__(self, algorithm: str, learning_rate: float) -> None:
       self.opt_algorithm = algorithm
       self.lr = learning_rate

Then in the config we would write:

.. code:: yaml

   optimiser:
       _target_: my_code.Optimizer
       algorithm: SGD
       learning_rate: 0.01

Finally to instantiate the object in the code from the config, we call:

.. code:: python

   optimiser = instantiate(config.optimiser)

and the object can now be used as normal.
