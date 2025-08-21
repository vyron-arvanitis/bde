"""this is a bde builder"""

from models.models import Fnn
from training.trainer import FnnTrainer


class BdeBuilder(Fnn, FnnTrainer):
    #TODO: build the BdeBuilderClass
    def __init__(self, sizes):
        Fnn.__init__(self, sizes)
        FnnTrainer.__init__(self)
        pass

    def get_model(self):
        pass

    def deep_ensemble_creator(self):
        pass

    def store_ensemble_results(self):
        pass

    def fit(self):
        pass
