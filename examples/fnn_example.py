import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import optax

from bde.models.models import Fnn
from bde.training.trainer import FnnTrainer
from bde.bde_builder import BdeBuilder
from bde.viz.plotting import plot_pred_vs_true
from bde.data.dataloader import DataLoader
from bde.data.preprocessor import DataPreProcessor
from bde.loss.loss import  LossMSE
from bde.sampler.mile_wrapper import MileWrapper

from bde.sampler.warmup import custom_mclmc_warmup
from bde.sampler.probabilistic import ProbabilisticModel
from bde.sampler.prior import Prior, PriorDist

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def main():
    # generate True data for test purposes
    # main_key = jax.random.PRNGKey(0)
    # k_X, k_W, k_eps = jax.random.split(main_key, 3)
    # X_true = jax.random.normal(k_X, (1024, 10))
    # true_W = jax.random.normal(k_W, (10, 1))
    # y_true = X_true @ true_W + 0.1 * jax.random.normal(k_eps, (1024, 1))

    # these steps should be inside the bde builder
    data = fetch_openml(name="airfoil_self_noise", as_frame=True)

    X = data.data.values   # shape (1503, 5)
    y = data.target.values.reshape(-1, 1)  # shape (1503, 1)

# Convert to JAX arrays
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert to JAX arrays
    X_train = jnp.array(X_train, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.float32)
    X_test = jnp.array(X_test, dtype=jnp.float32)
    y_test = jnp.array(y_test, dtype=jnp.float32)

    Xmu, Xstd = jnp.mean(X_train, 0), jnp.std(X_train, 0) + 1e-8
    Ymu, Ystd = jnp.mean(y_train, 0), jnp.std(y_train, 0) + 1e-8

    Xtr = (X_train - Xmu) / Xstd
    Xte = (X_test  - Xmu) / Xstd
    ytr = (y_train - Ymu) / Ystd
    yte = (y_test  - Ymu) / Ystd

    sizes = [5, 16, 16, 2] #TODO: for regression 2 nodes ( mean and var) and for classification user gives # of classes!

    # model = Fnn(hidden_layers)
    # trainer = FnnTrainer()
    # trainer.train(
    #     model=model,
    #     x=train_set.x,
    #     y=train_set.y,
    #     loss=LossMSE(),
    #     optimizer=trainer.default_optimizer(),  # the default optimizer!
    #     epochs=1000
    # )
    # print(trainer.history["train_loss"][:10])  # first 10 losses

    # y_pred = model.predict(test_set.x)
    # print("the first predictions are ", y_pred)



    print("-----------------------------------------------------------")
    bde = BdeBuilder(
        sizes, 
        n_members=1, 
        epochs=3000, 
        optimizer=optax.adam(1e-2)
        )
    
    print("Number of FNNs in the BDE: ", len(bde.members))
    
    # fit + predict
    bde.fit(
        x=Xtr, 
        y=ytr, 
        epochs=3000
        )
    
    initial_params = bde.all_fnns["fnn_0"]
    prior = PriorDist.STANDARDNORMAL.get_prior()
    model = ProbabilisticModel(module=bde.members[0], params=initial_params, prior=prior, n_batches=1)

    logdensity_fn = lambda params: model.log_unnormalized_posterior(params, x=Xtr, y=ytr)
    print(model.log_prior(initial_params))
    print(model.log_likelihood(initial_params, Xtr, ytr))

    warmup = custom_mclmc_warmup(
    logdensity_fn=logdensity_fn,
    diagonal_preconditioning=False,
    step_size_init=4e-3,
    desired_energy_var_start=0.5,
    desired_energy_var_end=0.1,
    trust_in_estimate=1.5,
    num_effective_samples=100,
    )
    
    rng_key = jax.random.PRNGKey(1)
    results = warmup.run(rng_key, position=initial_params, num_steps=1000)
    print("step_size:", results.parameters.step_size)
    print("L:", results.parameters.L)
    post_end = logdensity_fn(results.state.position)
    print("end-of-warmup post:", post_end)
    
    sampler = MileWrapper(logdensity_fn, step_size=results.parameters.step_size , L=results.parameters.L,)
    positions, infos, state = sampler.sample(rng_key=rng_key, init_position = results.state.position, num_samples = 5, thinning=10)
    
    fnn = bde.members[0]

    _, unravel = ravel_pytree(fnn.params)

    p = positions[-1]  
    if isinstance(p, jnp.ndarray):
        if p.ndim == 2:   
            p = unravel(p[-1])
        elif p.ndim == 1: 
            p = unravel(p)

    # print(p.shape) # AttributeError: 'tuple' object has no attribute 'shape'
    pred  = fnn.apply({'params': p}, Xte)
    mu_n  = pred[..., 0:1]
    sigma_n = 0.5 + 10.0 * jax.nn.sigmoid(pred[..., 1:2])

    y_pred = jnp.ravel(mu_n * Ystd + Ymu)
    y_err  = jnp.ravel(sigma_n * Ystd)
    y_true = jnp.ravel(yte * Ystd + Ymu)

    print("y_true shape:", y_true.shape, "y_pred shape:", y_pred.shape, "yerr shape:", y_err.shape)
    

########
    #print(bde_pred["ensemble_mean"])
    #print(bde_pred["ensemble_var"])

    #print("keys:", list(bde.keys()))  # ['ensemble_mean', 'ensemble_var']
    ##print(bde.ensemble_mean[:2]) # another way to access the ensemble_mean
    #print("mean shape:", bde_pred["ensemble_mean"].shape)
    #print("var shape:", bde_pred["ensemble_var"].shape)

    plot_pred_vs_true(
        y_pred=y_pred,
        y_true=y_true,
        y_pred_err=y_err,
        title="trial",
        savepath="to_be_deleted"
        )

if __name__ == "__main__":
    main()
