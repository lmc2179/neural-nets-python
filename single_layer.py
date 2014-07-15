import feedforward

class SingleLayerOnline(feedforward.FeedForward):
    def __init__(self, number_inputs):
        super(SingleLayerOnline, self).__init__()
        self.network_inputs = ['input_'+str(i) for i in range(number_inputs)]
        self.network_outputs = ['output']
        self.add_node('output', 'heaviside')
        [self.add_node(i, 'identity', outputs=self.network_outputs) for i in self.network_inputs]

    def get_all_weights(self):
        return [self.get_edge_weight(i,self.network_outputs[0]) for i in self.network_inputs]

    def fit(self, X, Y):
        [self._update(x,y) for x,y in zip(X,Y)]

    def _update(self, x, y):
        raise NotImplementedError

class Winnow(SingleLayerOnline):
    def _update(self, x, y):
        predicted_value = self.evaluate_pattern(x)[0]
        if predicted_value > y:
            [self._demote(index) for index, value in enumerate(x) if value == 1]
        if predicted_value < y:
            [self._promote(index) for index, value in enumerate(x) if value == 0]

    def _demote(self, weight_index):
        w = self.get_edge_weight(self.network_inputs[weight_index], self.network_outputs[0])
        self.set_edge_weight(self.network_inputs[weight_index], self.network_outputs[0], w/2.0)

    def _promote(self, weight_index):
        w = self.get_edge_weight(self.network_inputs[weight_index], self.network_outputs[0])
        self.set_edge_weight(self.network_inputs[weight_index], self.network_outputs[0], w*2.0)

class Perceptron(SingleLayerOnline):
    def __init__(self, number_inputs):
        super(Perceptron, self).__init__(number_inputs+1)

    def evaluate_pattern(self, input_vector):
        return super(Perceptron, self).evaluate_pattern([1]+list(input_vector))

    def _update(self, x, y):
        predicted_value = self.evaluate_pattern(x)[0]
        if predicted_value != y:
            coeff = 1 if predicted_value == 1 else -1
            weights = self.get_all_weights()
            inputs_with_bias = [1]+list(x)
            [self.set_edge_weight(input_name, self.network_outputs[0], w - coeff*inputs_with_bias[i])
             for i, (input_name, w) in enumerate(zip(self.network_inputs, weights))]

def test_winnow():
    net = Winnow(2)
    test_inputs = [(0, 1), (1, 1), (1, 0), (0, 0)] * 10
    test_outputs = [1, 1, 0, 0] * 10
    net.fit(test_inputs, test_outputs)
    print net.get_all_weights()

def test_perceptron():
    net = Perceptron(2)
    test_inputs = [(0, 1), (1, 1), (1, 0), (0, 0)] * 10
    test_outputs = [1, 1, 0, 0] * 10
    net.fit(test_inputs, test_outputs)
    print net.get_all_weights()

if __name__ == '__main__':
    test_winnow()
    test_perceptron()