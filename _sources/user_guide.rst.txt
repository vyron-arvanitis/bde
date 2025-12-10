.. title:: User guide : contents

.. _user_guide:

==========
User Guide
==========

This guide focuses on the pieces that are specific to ``bde``. If you are new to
scikit-learn's estimator API, refer to the official `developer guide
<https://scikit-learn.org/stable/developers/develop.html>`__ for the foundational
concepts. The sections below assume that background and concentrate on how
``BdeRegressor`` and ``BdeClassifier`` behave, how they integrate with JAX, and how
you should prepare data to get reliable results.
For installation, environment setup, and JAX device configuration, start with
:ref:`quick_start`.

Estimator overview
------------------

``bde`` exposes two scikit-learn compatible estimators:

* :class:`bde.BdeRegressor` for continuous targets.
* :class:`bde.BdeClassifier` for categorical targets.

Both inherit :class:`sklearn.base.BaseEstimator` and the relevant mixins, so they
support the familiar ``fit``/``predict``/``score`` methods, accept keyword
hyperparameters in ``__init__``, and can be dropped into a
:class:`sklearn.pipeline.Pipeline`. Under the hood they train a fully connected
ensemble in JAX and then run an MCMC sampler to draw posterior weight samples. At
prediction time the estimator combines those samples to provide means, standard
deviations, credible intervals, probability vectors, or the raw ensemble outputs.

You can customise the architecture and training stack: choose any activation
function, swap in your own optimiser, or rely on the defaults (``optax.adamw``).
Losses also default sensibly: regression uses :class:`bde.loss.GaussianNLL` and
classification uses :class:`bde.loss.CategoricalCrossEntropy`.

Data preparation
----------------

Bayesian deep ensembles are sensitive to feature and target scale because the
networks are initialised with zero-mean weights and the prior assumes unit-scale
activations. Large raw targets (for instance the default output of
:func:`sklearn.datasets.make_regression`) can lead to very poor fits if left
unscaled. Always apply basic preprocessing before calling ``fit``:

* Standardise features with :class:`sklearn.preprocessing.StandardScaler` (or an
  equivalent transformer) so each column has roughly zero mean and unit variance.
* For regression, standardise the target as well and keep the scaler handy if you
  need to transform predictions back to the original scale.


Gaussian likelihood (regression)
--------------------------------

Regression heads emit a mean and an unconstrained scale. The scale is mapped to a
positive standard deviation with ``softplus`` (plus a small epsilon) in all stages:
the training loss :class:`bde.loss.GaussianNLL <bde.loss.loss.GaussianNLL>`, the
posterior log-likelihood in :func:`bde.sampler.probabilistic.ProbabilisticModel.log_likelihood`,
and the prediction helpers in :func:`bde.bde_evaluator.BdePredictor._regression_mu_sigma`.

.. note::
   If you request ``raw=True`` from the regressor you receive the unconstrained scale
   head and should apply the same ``softplus`` transform before treating it as a
   standard deviation.


Understanding the outputs
-------------------------

The estimators expose several prediction modes:

``predict(X)``
    Returns the mean prediction (regression) or hard labels (classification).
``predict(X, mean_and_std=True)``
    Regression only; returns a tuple ``(mean, std)`` where ``std`` combines
    aleatoric and epistemic components.
``predict(X, credible_intervals=[0.05, 0.95])``
    Regression only; returns ``(mean, quantiles)`` where each quantile is computed
    from Monte Carlo samples drawn from every posterior component (i.e. the full
    mixture across ensemble members and MCMC draws). This reflects the predictive
    distribution of the entire ensemble rather than just parameter quantiles. For
    small posterior sample counts (``n_samples < 10``) a small random draw is used;
    for very large counts (``n_samples > 10_000``) a single sample is taken to keep
    the computation cheap.
``predict(X, raw=True)``
    Returns the raw tensor with leading axes ``(ensemble_members, samples, n,
    output_dims)``. Useful for custom diagnostics.
``predict_proba(X)``
    Classification only; returns class probability vectors.

How to read uncertainties
-------------------------

- **Mean + std** (``mean_and_std=True``): ``std`` is the total predictive standard deviation. It sums aleatoric variance (averaged scale head) and epistemic variance (spread of ensemble means), so high values mean either noisy data or disagreement across members.
- **Credible intervals** (``credible_intervals=[...])``): Quantiles are taken over *samples from the full mixture* of ensemble members and posterior draws. This captures both aleatoric and epistemic uncertainty. For example, requesting ``[0.05, 0.95]`` returns lower/upper curves you can treat as a 90% credible band.
- **Raw outputs** (``raw=True``): Shape ``(E, T, N, D)`` for regression where, ``E=ensemble_members``, ``T=n_samples``, ``N=n_data`` and ``D=2`` (mean, scale). You can manually compute aleatoric vs epistemic components, plot per-member predictions, or customise intervals if needed.


Key hyperparameters
-------------------

**Model architecture**

- ``n_members``
    Number of deterministic networks in the ensemble. Increasing members improves
    epistemic uncertainty estimation but raises computational cost (if enough
    parallel devices are available training time is not affected).
- ``hidden_layers``
    Widths of hidden layers. Defaults internally to ``[4, 4]`` if ``None``.

**Pre-sampling optimization**

- ``epochs`` / ``patience``
    Control how long the deterministic pre-training runs before sampling. ``epochs``
    is the hard cap; ``patience`` triggers early stopping when the validation loss
    plateaus so the sampler starts from a high-likelihood region. When ``patience``
    is ``None`` training always runs for all epochs.
- ``lr``
    Learning rate for the Adam optimiser during pre-sampling training.

**Sampling**

- ``warmup_steps`` / ``n_samples`` / ``n_thinning``
    Control the MCMC sampling stage. ``warmup_steps`` adjusts the step size,
    ``n_samples`` defines the number of retained posterior draws, and
    ``n_thinning`` specifies the interval between saved samples.

- ``desired_energy_var_start`` / ``desired_energy_var_end`` / ``step_size_init``
    Configure the samplers behaviour. The ``desired_energy_var_*`` parameters
    set the target variance of the energy during sampling which is linearly
    annealed from start to end over the course of the warmup phase. The
    ``step_size_init`` parameter sets the initial step size for the dynamics
    integrator; this is adapted during warmup to reach the desired energy
    variance. For medium sized BNNs a good default is to set
    ``desired_energy_var_start=0.5``, ``desired_energy_var_end=0.1``, and
    pick the learning rate as the ``step_size_init`` (or slightly larger). For simpler
    models or highly overparameterized settings (for example a 2x16 network provides
    good results on a small dataset, then using a 3x32 network would be considered
    highly overparameterized) decreasing the desired energy variance targets might
    be necessary to reach good performance. **The desired energy variance is the most
    important hyperparameter** to tune for sampler performance.

Sampler and builder internals
-----------------------------

After the deterministic training phase ``BdeRegressor`` and ``BdeClassifier``
construct a :class:`bde.bde_builder.BdeBuilder` instance. This helper manages the
ensemble members, coordinates parallel training across devices, and hands off to
``bde.sampler`` utilities for warmup and sampling. Advanced users can interact
with these pieces directly:

* ``estimator._bde`` references the builder after ``fit`` and exposes the
  deterministic members and training history.
* ``estimator.positions_eT_`` stores the weight samples with shape ``(E, T, ...)``.

Generally you should rely on the high-level estimator API, but the internals are
accessible for custom diagnostics or research experiments.

Where to next
-------------

* The :ref:`quick_start` page shows condensed scripts you can run end to end.
* :ref:`api` documents every public class and helper in the package.
* :ref:`general_examples` renders notebooks and plots that mirror the examples
  in the ``examples/`` directory.
