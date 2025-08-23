"""This class is for preprocessing data"""

from .dataloader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


class DataPreProcessor:
    def __init__(self, data: DataLoader):
        self.data = data

    def norm_data(self, data: DataLoader) -> DataLoader:
        # TODO: code [later]

        pass

    def aug_data(self, data: DataLoader) -> DataLoader:
        # TODO: code [later]

        pass

    def add_noise_data(self, data: DataLoader) -> DataLoader:
        # TODO: code [later]

        pass

    def split(self, test_size=0.15, val_size=0.15, random_state=42, stratify=False):
        N = len(self.data)
        idx = np.arange(N)
        strat = (np.ravel(np.array(self.data.y)) if (stratify and self.data.y is not None) else None)

        trval_idx, te_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=strat
        )

        val_rel = val_size / (1 - test_size) if (1 - test_size) > 0 else 0.0
        strat_trval = (np.ravel(np.array(self.data.y))[trval_idx] if strat is not None else None)
        tr_idx, val_idx = train_test_split(
            trval_idx, test_size=val_rel, random_state=random_state, stratify=strat_trval
        )

        # thanks to DataLoader.__getitem__, this returns subset loaders
        train_loader = self.data[tr_idx]
        val_loader = self.data[val_idx]
        test_loader = self.data[te_idx]
        return train_loader, val_loader, test_loader
