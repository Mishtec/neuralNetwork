# Model
class Model:
    def __init__(self):
        self.dataset = None

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_dataset(self):
        return self.dataset
