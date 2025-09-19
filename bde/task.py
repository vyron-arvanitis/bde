from enum import Enum
from bde.loss.loss import *


class TaskType(Enum):
    """Holder for task types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

    def validate_loss(self, loss: BaseLoss):
        """
        Validate that the given loss is compatible with the current task.

        Parameters
        ----------
        loss : BaseLoss
            The loss to be validated

        Raises
        ------
        ValueError
            If the provided loss is not permitted for this task type.

        """

        allowed = {
            TaskType.REGRESSION: {GaussianNLL, Rmse},
            TaskType.CLASSIFICATION: {BinaryCrossEntropy, CategoricalCrossEntropy},
        }
        if loss.__class__ not in allowed[self]:
            raise ValueError(
                f"{loss.__class__.__name__} not allowed for task {self.value}. "
                f"Allowed: {[cls.__name__ for cls in allowed[self]]}"
            )
