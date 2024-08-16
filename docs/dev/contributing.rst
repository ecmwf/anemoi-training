##############
 Contributing
##############

To contribute, you first need to clone the repository from
https://github.com/ecmwf/anemoi-training/ and then install as follows:

.. code:: bash

   # to install all dependencies
   pip install -e .
   # to install dependencies for code development
   pip install -e '.[dev]'

To build the documentation you may also have to install pandoc on macOS:

.. code:: bash

   brew install pandoc

**********************
 Pre-Commit Etiquette
**********************

Please use pre-commit hooks. You can find the config in
`.pre-commit-config.yaml`, which automatically format new code and check
with tools like `black` and `flake8`.

When you first set up this repo, run:

.. code:: bash

   pre-commit install

This will install the pre-commit hooks. To ensure that the hooks are
working, run:

.. code:: bash

   pre-commit run --all-files

This will download the required tools and run the hooks on all files in
the repository.

*********
 Commits
*********

Please ensure each commit message is informative and concise. Remember
that it is better to make small changes and commit frequently. This
makes it easier to track changes and revert if necessary.

.. note::

   Please don't push changes directly to `main`. Instead, pull-request
   changes from your own branch into `origin/develop` so they get
   peer-reviewed.

*************
 How to test
*************

We have written tests using the `pytest` functional interface. They're
stored in the `tests` directory. With the developer dependencies
installed you can simply run

.. code:: bash

   pytest

or if you just want to run a specific file, run:

.. code:: bash

   pytest tests/test_<file>.py

Be aware that some tests like the `test_gnn.py` run a singular forward
pass, which can be slow on CPU and runs better on GPU.

***************
 Documentation
***************

We use Sphinx to generate the documentation. To build the documentation:

.. code:: bash

   cd docs
   make html

This will generate the documentation in the `docs/_build/html`
directory. Open `docs/_build/html/index.html` to view the documentation.

..
   How to Profile

..
   ==============

..
   We wrote a special profiler that uses PyTorch, Lightning, and memray to

..
   measure the performance of the code in it's current training state. Run

..
   .. code:: bash

..
   anemoi-traing profile

..
   This starts a short training run and creates different information:

..
   -  Time Profile: Duration of different operations

..
   -  Speed Profile: Throughput of dataloader and model

..
   -  Memory Profile: Memory of the "worst offenders"

..
   -  System Utilization: Overall system utilization (needs W&B online)
