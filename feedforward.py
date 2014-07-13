import networkx

class FeedForward(object):
    def __init__(self):
        self.network_inputs = None
        self.network_outputs = None
        self.network = networkx.DiGraph()

    def add_node(self, name, activation_name, inputs=None, outputs=None):
        self.network.add_node(name)
        self.network[name]['activation_function'] = self._activation_function_factory(activation_name)
        if inputs is not None:
            [self.network.add_edge(i, name, w=self._get_default_edge_weight()) for i in inputs]
        if outputs is not None:
            [self.network.add_edge(name, o, w=self._get_default_edge_weight()) for o in outputs]

    def _get_default_edge_weight(self):
        return 1.0

    def _activation_function_factory(self, name):
        if name == 'identity':
            return lambda x:x
        # TODO: Add sigmoid and arctan

    def set_edge_weight(self, a, b, weight):
        self.network.edge[a][b]['w'] = weight

    def get_edge_weight(self, a, b):
        return self.network[a][b]['w']

    def evaluate_pattern(self, input_vector):
        known_node_values = dict(zip(self.network_inputs,input_vector))
        output_vector = [self._calculate_node_value(o, known_node_values) for o in self.network_outputs]
        return output_vector

    def _calculate_node_value(self, node, known_node_values):
        if node in known_node_values:
            return known_node_values[node]
        input_nodes = self.network.predecessors(node)
        input_weights_and_values = [(self.get_edge_weight(i,node), self._calculate_node_value(i, known_node_values)) for i in input_nodes]
        summed_inputs = sum(w*i for w, i in input_weights_and_values)
        return self.network[node]['activation_function'](summed_inputs)

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

test_calculate_node_single_layer()
test_calculate_node_hidden_layer()