# Controller
class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def update_dataset(self, dataset):
        self.model.set_dataset(dataset)
        self.view.display_dataset(self.model.get_dataset())

