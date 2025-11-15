"delete before release"

import jax.numpy as jnp


# check lppd
class metrics:
    def __init__(self, y_true, y_pred, sigma):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sigma = sigma

    def mae(self):
        return jnp.mean(jnp.abs(self.y_true - self.y_pred))

    def rmse(self):
        return jnp.sqrt(jnp.mean((self.y_true - self.y_pred) ** 2))

    def predictive_accuracy(self, sigma1=True, sigma2=False, sigma3=False):
        # Pulls
        pulls = (self.y_true - self.y_pred) / self.sigma
        pull_mean = jnp.mean(pulls)
        pull_std = jnp.std(pulls)
        outs = {
            "pull_mean": float(pull_mean),
            "pull_std": float(pull_std),
        }

        # Coverage
        flags = {1: sigma1, 2: sigma2, 3: sigma3}
        for i, enabled in flags.items():
            if enabled:
                cov = jnp.mean(jnp.abs(self.y_true - self.y_pred) <= i * self.sigma)
                outs[f"coverage_{i}σ"] = float(cov)

        return outs
