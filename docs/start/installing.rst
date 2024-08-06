############
 Installing
############

To install the package, you can use the following command:

.. code:: bash

   pip install anemoi-training

**Please don't push changes directly to `main`**. Instead, PR changes from your own branch into `origin/main` so they get peer-reviewed.

**************
 Contributing
**************

To contribute, you first need to clone the repository from https://github.com/ecmwf/anemoi-training/ and then install as follows:

.. code:: bash

   # to install all dependencies
   pip install -e .
   # to install dependencies for code development
   pip install -e .[dev]

You may also have to install pandoc on MacOS:

.. code:: bash

   brew install pandoc

Pre-Commit Etiquette
--------------------

Please use pre-commit hooks. You can find the config in `.pre-commit-config.yaml`, which automatically format new code and check with tools like `black` and `flake8`.

When you first set up this repo, run:

.. code:: bash

  pre-commit install

to enable these code formatters.


How to test
-----------
We have written tests using the `pytest` functional interface.

They're stored in the tests/ directory. After installing `pytest` (`pip install pytest`) you can simply run

.. code:: bash

  pytest

or if you just want to run a specific file, run:

.. code:: bash

  pytest tests/test_<file>.py


Be aware that some tests like the `test_gnn.py` run a singular forward pass, which can be slow on CPU and runs better on GPU.

How to Profile
--------------

We wrote a special profiler that uses Pytorch, Lightning, and memray to measure the performance of the code in it's current training state. Run

.. code:: bash

  aifs-profile

This starts a short training run and creates different information:

- Time Profile: Duration of different operations
- Speed Profile: Throughput of dataloader and model
- Memory Profile: Memory of the "worst offenders"
- System Utilization: Overall system utilization (needs W&B online)
