.. _quick_start:

###############
Quick start
###############

`bde` implements Bayesian Deep Ensembles that plug directly into scikit-learn
pipelines while running training and sampling in JAX. This page walks through
setting up the environment, running your first estimators, and validating the
installation.

Installation
============

The package will be published on PyPI soon. Until then you can install the
current source build:

.. prompt:: bash $

  pip install git+https://github.com/scikit-learn-contrib/bde.git

If you prefer reproducible environments with GPU or multi-device CPU support,
consider using `pixi <https://prefix.dev/docs/pixi/overview>`__ as described in
:ref:`user_guide`. The repo already ships with a ``pixi.toml`` that pins the
required versions of JAX, scikit-learn, and the CUDA toolchain when available.

Set the JAX device count
========================

When you run the estimators outside of the examples shipped in ``examples/``,
ensure JAX can see enough host devices. The environment variable below makes
CPU-only runs allocate eight virtual devices; tweak the value to match your
hardware:

.. prompt:: bash $

  export XLA_FLAGS="--xla_force_host_platform_device_count=8"

You can set it once per shell session or inject it programmatically before
importing JAX in your Python scripts.

Work with the development environment
=====================================

Once you clone the repository, bootstrap the dev dependencies and tooling with:

.. prompt:: bash $

  pixi install

You can then run the standard tasks:

* ``pixi run lint`` for style checks (``ruff`` and ``black``)
* ``pixi run test`` for the pytest suite, including scikit-learn compatibility
  checks in ``tests/test_common.py``
* ``pixi run build-doc`` to render this documentation locally with Sphinx and
  sphinx-gallery examples

If you prefer calling Python tooling directly, enter the environment shell:

.. prompt:: bash $

  pixi shell -e dev

From there ``pytest``, ``ruff`` and ``sphinx-build`` are available on ``PATH``.

Datasets included in the package for testing purposes
=====================================================

Airfoil
-------

Airfoil Self-Noise Dataset
Source: UCI Machine Learning Repository (Dua & Graff, 2017).
Task: Regression (predicting scaled sound pressure level).
Notes: This is the standard ``airfoil_self_noise`` dataset used in many regression benchmarks.

Iris
----

Iris Dataset
Source: Fisher (1936); canonical modern version distributed via scikit-learn.
Task: Multiclass classification (setosa, versicolor, virginica).
Notes: Standard toy dataset for testing classification models.

Concrete
--------

Concrete Compressive Strength Dataset
Source: UCI Machine Learning Repository (Yeh, 2006).
Task: Regression (predicting concrete compressive strength based on mixture components).
Notes: Widely used as a tabular regression benchmark. Your file ``concrete.data`` matches the UCI format.


Next steps
==========

* Dive into :ref:`user_guide` for a walkthrough of the estimator internals and
  configuration knobs.
* Explore the auto-generated API reference in :ref:`api` to see every public
  class, function, and dataclass.
* Run the gallery in :ref:`general_examples` to compare regression and
  classification behaviours on real-world datasets.
