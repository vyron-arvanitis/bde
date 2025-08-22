"""This class is for preprocessing data"""

from dataloader import DataLoader


class DataPreProcessor:
    def __init__(self, data: DataLoader):
        self.data = data

    def norm_data(self, data: DataLoader) -> DataLoader:
        #TODO: code

        pass

    def aug_data(self, data: DataLoader) -> DataLoader:
        #TODO: code

        pass

    def add_noise_data(self, data: DataLoader) -> DataLoader:
        #TODO: code

        pass

    def split(self):
        """This method splits the data train, validate, test or train test
        Returns
        -------

        """
        #TODO: code
        pass