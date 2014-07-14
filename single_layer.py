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


def test_winnow():
    net = Winnow(2)
    test_inputs = [(0, 1), (1, 1), (1, 0), (0, 0)] * 10
    test_outputs = [1, 1, 0, 0] * 10
    net.fit(test_inputs, test_outputs)
    print net.get_all_weights()

if __name__ == '__main__':
    test_winnow()