import networkx
import math

class FeedForward(object):
    def __init__(self):
        self.network_inputs = None
        self.network_outputs = None
        self.network = networkx.DiGraph()

    def add_node(self, name, activation_name, inputs=None, outputs=None):
        self.network.add_node(name, activation_function=self._activation_function_factory(activation_name))
        # self.network[name]['activation_function'] = self._activation_function_factory(activation_name)
        if inputs is not None:
            [self.network.add_edge(i, name, w=self._get_default_edge_weight()) for i in inputs]
        if outputs is not None:
            [self.network.add_edge(name, o, w=self._get_default_edge_weight()) for o in outputs]

    def _get_default_edge_weight(self):
        return 1.0

    def _activation_function_factory(self, name):
        if name == 'identity':
            return lambda x:x
        if name == 'heaviside':
            return lambda  x: 1 if x >= 0 else 0
        if name == 'logistic':
            return lambda x: 1.0/(1.0 + math.exp(-x))
        if name == 'none':
            return lambda x: None

    def _logistic(self, x):
        return 1.0/(1.0 + math.exp(-x))

    def _logistic_derivative(self, x):
        return self._logistic(x) * (1 - self._logistic(x))

    def set_edge_weight(self, a, b, weight):
        self.network.add_edge(a,b,w=weight)

    def get_edge_weight(self, a, b):
        return self.network[a][b]['w']

    def evaluate_pattern(self, input_vector):
        known_node_values = dict(zip(self.network_inputs,input_vector))
        output_vector = [self._calculate_node_value(o, known_node_values) for o in self.network_outputs]
        return output_vector


    def _calculate_node_value(self, node, known_node_values):
        if node in known_node_values:
            return known_node_values[node]
        summed_inputs = self._get_neuron_summed_inputs(known_node_values, node)
        return self.network.node[node]['activation_function'](summed_inputs)

    def _get_neuron_summed_inputs(self, node, known_node_values):
        input_nodes = self.network.predecessors(node)
        input_weights_and_values = [(self.get_edge_weight(i, node), self._calculate_node_value(i, known_node_values))
                                    for i in input_nodes]
        summed_inputs = sum(w * i for w, i in input_weights_and_values)
        return summed_inputs

def test_calculate_node_single_layer():
    net = FeedForward()
    net.network_inputs = ['input 1', 'input 2']
    net.network_outputs = ['output']
    net.add_node('input 1', 'identity')
    net.add_node('input 2', 'identity')
    net.add_node('output', 'identity', inputs=['input 1', 'input 2'])
    net.set_edge_weight('input 1', 'output', 1.0)
    net.set_edge_weight('input 2', 'output', 1.0)
    result = net.evaluate_pattern([2.0, 2.0])
    print result
    assert result == [4.0]

def test_calculate_node_hidden_layer():
    net = FeedForward()
    net.network_inputs = ['input 1', 'input 2']
    net.network_outputs = ['output']
    net.add_node('input 1', 'identity')
    net.add_node('input 2', 'identity')
    net.add_node('hidden 1', 'identity', inputs=['input 1'])
    net.add_node('output', 'identity', inputs=['hidden 1', 'input 2'])
    net.set_edge_weight('input 1', 'hidden 1', 1.0)
    net.set_edge_weight('hidden 1', 'output', 1.0)
    net.set_edge_weight('input 2', 'output', 1.0)
    result = net.evaluate_pattern([2.0, 2.0])
    print result
    assert result == [4.0]

if __name__ == '__main__':
    test_calculate_node_single_layer()
    test_calculate_node_hidden_layer()