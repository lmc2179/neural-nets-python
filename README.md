neural-nets-python
==================

An implementation of a variety of neural networks in python.

It is currently quite sparse. The following classes have been implemented:

1. feedforward.py
  * FeedForward neural network (Abstract)
2. single_layer.py:
  * SingleLayerOnline: Single-layer, feedforward networks with a learning rule (Abstract).
  * Perceptron: The famous two-class [Perceptron](http://en.wikipedia.org/wiki/Perceptron)
  * Winnow: Littlestone's [Winnow algorithm](http://en.wikipedia.org/wiki/Winnow_(algorithm) for binary features.
3. multilayer_perceptron.py
  * Multilayer: Feed-forward multilayer perceptron with sigmoidal hidden unit activation functions (Abstract)