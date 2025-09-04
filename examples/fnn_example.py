import jax
import jax.numpy as jnp
import optax

from bde.models.models import Fnn
from bde.training.trainer import FnnTrainer
from bde.bde_builder import BdeBuilder
from bde.viz.plotting import plot_pred_vs_true
from bde.data.dataloader import DataLoader
from bde.data.preprocessor import DataPreProcessor
from bde.loss.loss import  LossMSE

from bde.sampler.warmup import custom_mclmc_warmup
from bde.sampler.probabilistic import ProbabilisticModel

from bde.sampler.prior import Prior, PriorDist

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
    data = DataLoader(seed=43, n_samples=500, n_features=10)  # creates automatically the gen data
    prep = DataPreProcessor(data)
    train_set, val_set, test_set = prep.split() #TODO: we only need validation

    sizes = [10, 64, 64, 1] #TODO: for regression 2 nodes ( mean and var) and for classification user gives # of classes!

    # model = Fnn(sizes)
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
        epochs=100, 
        optimizer=optax.adam(1e-5)
        )
    
    print("Number of FNNs in the BDE: ", len(bde.members))
    
    # fit + predict
    bde.fit(
        x=train_set.x, 
        y=train_set.y, 
        epochs=100
        )
    
    initial_params = bde.all_fnns["fnn_0"]
    prior = PriorDist.STANDARDNORMAL.get_prior()
    model = ProbabilisticModel(module=bde.members[0], params=initial_params, prior=prior, n_batches=1)

    logdensity_fn = lambda params: model.log_unnormalized_posterior(params, x=train_set.x, y=train_set.y)

    #bde_pred = bde.predict_ensemble(test_set.x, include_members=True)

    warmup = custom_mclmc_warmup(
    logdensity_fn=logdensity_fn,
    diagonal_preconditioning=True,
    step_size_init=1e-5,
    desired_energy_var_start=1e-3,
    desired_energy_var_end=1e-6,
    trust_in_estimate=1.5,
    num_effective_samples=100,
)
    

    
    rng_key = jax.random.PRNGKey(1)
    results = warmup.run(rng_key, position=initial_params, num_steps=1000)
    print("step_size:", results.parameters.step_size)
    print("L:", results.parameters.L)


########
    # print(bde_pred["ensemble_mean"])c
    # print(bde_pred["ensemble_var"])

    # print("keys:", list(bde.keys()))  # ['ensemble_mean', 'ensemble_var']
    # #print(bde.ensemble_mean[:2]) # another way to access the ensemble_mean
    # print("mean shape:", bde_pred["ensemble_mean"].shape)
    # print("var shape:", bde_pred["ensemble_var"].shape)

    # plot_pred_vs_true(
    #     y_pred=bde_pred["ensemble_mean"],
    #     y_true=test_set.y,
    #     y_pred_err=bde_pred["ensemble_var"],
    #     title="trial",
    #     savepath="to_be_deleted"
    #     )

if __name__ == "__main__":
    main()
