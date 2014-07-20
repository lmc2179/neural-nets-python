import multilayer_perceptron

class BackpropNetwork(multilayer_perceptron.Multilayer):
    """
    Backpropagation network with one layer of hidden units, sigmoidal hidden unit activations,
    and identity output unit activations.
    """
    def learn_pattern(self, input_vector, output_values):
        node_values = self._forward_propagate(input_vector)
        deltas = self._get_deltas(node_values, output_values)
        self._adjust_weights(node_values, deltas)

    def _forward_propagate(self, input_vector):
        known_values = dict(zip(self.network_inputs, input_vector))
        for layer in self.hidden_layers+[self.network_outputs]:
            for unit in layer:
                known_values[unit] = self._calculate_node_value(unit, known_values)
        return known_values

    def _get_deltas(self, node_values, output_values):
        deltas = dict([node_values[o] - output_values[i] for i,o in enumerate(self.network_outputs)])
        for hidden_layer in reversed(self.hidden_layers):
            for unit in hidden_layer:
                deltas[unit] = self._calculate_delta(unit, deltas, node_values)
        return deltas

    def _calculate_delta(self, node, deltas, node_values):
        summed_inputs = self._get_neuron_summed_inputs(node, node_values)
        summed_weighted_deltas = sum([self.get_edge_weight(node, o)*deltas[o] for o in self.network.successors(node)])
        return self._logistic_derivative(summed_inputs) * summed_weighted_deltas

    def _adjust_weights(self, node_values, deltas):
        for n1, n2 in self.network.edges():
            w = self.get_edge_weight(n1, n2)
            self.set_edge_weight(n1, n2, w - (node_values[n1]*deltas[n2]))