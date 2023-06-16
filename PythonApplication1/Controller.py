# Controller
class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def update_dataset(self, dataset):
        self.model.set_dataset(dataset)
        self.model.set_inputs()
        self.model.set_outputs()
        self.view.display_dataset(self.model.get_dataset())
        self.view.display_inputs(self.model.get_inputs())
        self.view.display_inputs(self.model.get_outputs())

    def print_weights(self):
        self.view.display_inputs(self.model.get_weights())

    def print_biases(self):
        self.view.display_inputs(self.model.get_biases())

    def update_biases_wieghts(self, weights, biases):
        self.model.set_weights(weights)
        self.model.set_biases(biases)
        self.view.display_inputs(self.model.get_weights())
        self.view.display_inputs(self.model.get_biases())


        


    

