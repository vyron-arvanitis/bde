from enum import Enum
from loss.loss import *


class TaskType(Enum):
    """Holder for task types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

    def validate_loss(self, loss: BaseLoss):
        allowed = {
            TaskType.REGRESSION: {GaussianNLL},
            TaskType.CLASSIFICATION: {BinaryCrossEntropy, CategoricalCrossEntropy},
        }
        if loss.__class__ not in allowed[self]:
            raise ValueError(
                f"{loss.__class__.__name__} not allowed for task {self.value}. "
                f"Allowed: {[cls.__name__ for cls in allowed[self]]}"
            )


