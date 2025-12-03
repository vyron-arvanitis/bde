---
title: 'bde: A Python Package for Bayesian Deep Ensembles via MILE'
tags:
  - Python
  - machine learning
  - MCMC
  - Bayesian deep learning
  - uncertainty quantification
authors:
  - name: Vyron Arvanitis
    equal-contrib: true
    affiliation: 1
  - name: Angelos Aslanidis
    equal-contrib: true
    affiliation: 1
  - name: Emanuel Sommer
    orcid: 0000-0002-1606-7547
    equal-contrib: true
    corresponding: true
    affiliation: "2, 3"
    email: emanuel.sommer@stat.uni-muenchen.de
  - name: David Rügamer
    orcid: 0000-0002-8772-9202
    affiliation: 3
affiliations:
 - name: Faculty of Physics, LMU Munich, Munich, Germany
   index: 1
 - name: Department of Statistics, LMU Munich, Munich, Germany
   index: 2
 - name: Munich Center for Machine Learning, Munich, Germany
   index: 3
date: 10 December 2025
bibliography: paper.bib

---

# Summary

`bde` is a Python package designed to bring state-of-the-art sampling-based Bayesian Deep Learning (BDL) to practitioners and researchers. The package combines the speed and high-performance capabilities of JAX [@jax2018github] with the user-friendly API of scikit-learn [@scikit-learn]. It specifically targets tabular supervised learning tasks, including distributional regression and (multi-class) classification, providing a seamless interface for Bayesian Deep Ensembles (BDEs) [@sommer2024connecting] via **Microcanonical Langevin Ensembles (MILE)** [@sommer2025mile].

The workflow of `bde` implements the robust two-stage BDE inference process of MILE. First, it optimizes `n_members` many (usually 8) independent, flexibly configurable feed-forward neural networks using regularized empirical risk minimization (with the negative log-likelihood as loss) via the AdamW optimizer [@loshchilov2018decoupled]. Second, it transitions to a sampling phase using Microcanonical Langevin Monte Carlo [@robnik2023microcanonical; @robnik2024fluctuation], enhanced with a tuning phase adapted for Bayesian Neural Networks [@sommer2025mile]. In essence optimization finds diverse high-likelihood modes; sampling explores local posterior structure. This process generates an ensemble of samples (models) that constitute an implicit posterior approximation.

Because optimization and sampling across ensemble members are independent, bde exploits JAX’s parallelization and just-in-time compilation to scale efficiently across CPUs, GPUs, and TPUs. Given new test data, the package approximates the posterior predictive, enabling point predictions, credible intervals, coverage estimates, and other uncertainty metrics through a unified interface.

# Statement of Need

Reliable uncertainty quantification (UQ) is increasingly viewed as a critical component of modern machine learning systems, and Bayesian Deep Learning provides a principled framework for achieving it [@papamarkou2024position]. While several libraries support optimization-based approaches such as variational inference or classical Bayesian modeling, accessible tools for sampling-based inference in Bayesian neural networks remain scarce. Existing probabilistic programming frameworks offer general-purpose MCMC but require substantial manual configuration to achieve competitive performance on neural network models.

`bde` addresses this gap by providing the first user-friendly implementation of MILE-a hybrid sampling technique shown to deliver strong predictive accuracy and calibrated uncertainty for Bayesian neural networks [@sommer2025mile]. By providing full scikit-learn compatibility, the package enables seamless integration into existing machine learning workflows, allowing users to obtain principled Bayesian uncertainty estimates without specialized knowledge of MCMC dynamics, initialization strategies, or JAX internals.

Through automated orchestration of optimization, sampling, parallelization, and predictive inference, `bde` offers a fast, reproducible, and practical solution for applying sampling-based BDL methods to tabular supervised learning tasks.

# Usage Example

TBD

## Regression

for example airfoil - potentially benchmark against other methods like random forests, deep ensembles, xgboost, tabpfn?

TBD (also pipeline with hyperparameter tuning?)

## Classification

TBD (not necessary if regression example is sufficient and verbose enough - note that article should not exceed 1000 words!)

# Acknowledgements

TBD

# References
