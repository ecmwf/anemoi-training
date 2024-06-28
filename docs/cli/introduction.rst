Introduction
============

When you install the `anemoi-training` package, this will also install command line tool
called ``anemoi-training`` which can be used to train models.

The tool can provide help with the ``--help`` options:

.. code-block:: bash

    % anemoi-training --help

The commands are:

.. toctree::
    :maxdepth: 1

    train

.. argparse::
    :module: anemoi.training.__main__
    :func: create_parser
    :prog: anemoi-training
    :nosubcommands:
