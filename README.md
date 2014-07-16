neural-nets-python
==================

An implementation of a variety of neural networks in python.

It is currently quite sparse. The following classes have been implemented:
0. feedforward.py
  * FeedForward neural network (Abstract)
0. single_layer.py:
  * SingleLayerOnline: Single-layer, feedforward networks with a learning rule (Abstract).
  * Perceptron: The famous two-class [Perceptron](http://en.wikipedia.org/wiki/Perceptron)
  * Winnow: Littlestone's [Winnow algorithm](http://en.wikipedia.org/wiki/Winnow_(algorithm) for binary features.
0. multilayer_perceptron.py
  * Multilayer: Feed-forward multilayer perceptron with sigmoidal hidden unit activation functions (Abstract)