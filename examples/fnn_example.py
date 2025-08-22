import jax
import jax.numpy as jnp
import optax

from bde.models.models import Fnn
from bde.training.trainer import FnnTrainer
from bde.bde_builder import BdeBuilder
from bde.viz.plotting import plot_pred_vs_true
from bde.data.dataloader import DataLoader
from bde.data.preprocessor import DataPreProcessor

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


    data = DataLoader(seed=43, n_samples=500, n_features=10) # creates automatically the gen data
    prep =  DataPreProcessor(data)
    train_set, val_set, test_set = prep.split()



    sizes = [10, 64, 64, 1]

    model = Fnn(sizes)
    trainer = FnnTrainer()
    trainer.fit(
        model=model,
        x=data.x,
        y=data.y,
        optimizer=trainer.default_optimizer(), #the default optimizer!
        epochs=1000
    )
    print(trainer.history["train_loss"][:10])  # first 10 losses


    y_pred = model.predict(data.x)
    # print("the first predictions are ", y_pred)


    print("-----------------------------------------------------------")
    bde = BdeBuilder(sizes, n_members=3, epochs=500, optimizer=optax.adam(1e-2))
    print(bde)
    # fit + predict
    bde.fit(x=data.x, y=data.y, optimizer=bde.optimizer, epochs=500, model=None)
    out = bde.predict_ensemble(data.x, include_members=True)
    # print(out["ensemble_mean"])
    # print(out["ensemble_var"])

    print("keys:", list(out.keys()))            # ['ensemble_mean', 'ensemble_var']
    print("mean shape:", out["ensemble_mean"].shape) # TODO: I THINK THERE IS AN ISSUE HERE! WITH THE SHAPE (500,1)
    plot_pred_vs_true(out["ensemble_mean"], data.y, "trial", savepath="to_be_deleted" )

if __name__ == "__main__":
    main()