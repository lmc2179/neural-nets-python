import feedforward

class Multilayer(feedforward.FeedForward):
    def __init__(self, input_sizes):
        """Takes a list of integers and constructs a multilayer network where the size of the nth layer
        is the nth integer in the list. Each node is connected to every node in the next layer.
        The first number is the number of inputs, the last number is the number of outputs."""
        if len(input_sizes) < 3:
            raise Exception('Expected at least three or more layers of neurons')
        super(Multilayer, self).__init__()
        self._initialize_neurons(input_sizes)

    def _initialize_neurons(self, input_sizes):
        self._initialize_layer_names(input_sizes)
        self._initialize_nodes()
        self._initialize_edges()

    def _initialize_edges(self):
        self._connect_layers(self.network_inputs, self.hidden_layers[0])
        self._connect_layers(self.hidden_layers[-1], self.network_outputs)
        for back_layer, front_layer in zip(self.hidden_layers[:-1], self.hidden_layers[1:]):
            self._connect_layers(back_layer, front_layer)

    def _initialize_nodes(self):
        [self.add_node(i, 'none') for i in self.network_inputs]
        [self.add_node(o, 'identity') for o in self.network_outputs]
        [self.add_node(h, 'logistic') for hidden_layer in self.hidden_layers for h in hidden_layer]

    def _connect_layers(self, layer_1, layer_2):
        from itertools import product
        [self.set_edge_weight(a,b, self._get_default_edge_weight()) for a,b in product(layer_1, layer_2)]

    def _initialize_layer_names(self, input_sizes):
        number_inputs, hidden_layers, number_outputs = input_sizes[0], input_sizes[1:-1], input_sizes[-1]
        self.network_inputs = ['input_' + str(i) for i in range(number_inputs)]
        self.network_outputs = ['output_' + str(i) for i in range(number_inputs)]
        self.hidden_layers = [['hidden_layer_' + str(i) + '_unit_' + str(h) for h in range(layer)] for i, layer in
                              enumerate(hidden_layers)]

def test_multilayer():
    net = Multilayer([2,5,1])
    print net.network_inputs
    print net.network_outputs
    print net.hidden_layers
    print net.network.edges()
    print net.evaluate_pattern([1.0, 1.0])
    import matplotlib.pyplot as plt
    import networkx
    networkx.draw_spring(net.network)
    plt.show()


if __name__=='__main__':
    test_multilayer()

