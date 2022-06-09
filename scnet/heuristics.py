import tensorflow as tf

from abc import ABC as AbstractBaseClass
from abc import abstractmethod

from scnet.graph_utils import get_roots_batched


class Heuristic(AbstractBaseClass):

    @abstractmethod
    def build(self, properties, num_classes):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, y):
        raise NotImplementedError()


class PreservePrediction(Heuristic):

    def build(self, properties, num_classes):
        self._roots = tf.transpose(
            tf.concat(
                [
                    get_roots_batched(p.post.make_graph(num_classes)) 
                    for p in properties
                ],
                axis=0))

        self._num_classes = num_classes

    def __call__(self, y):
        return tf.argsort(
            tf.one_hot(tf.argmax(y, axis=1), self._num_classes) @ self._roots,
            axis=1)


class SoftPreservePrediction(PreservePrediction):

    def __call__(self, y):
        return tf.argsort(tf.nn.softmax(y) @ self._roots, axis=1)


class NoHeuristic(Heuristic):

    def build(self, properties, num_classes):
        disjunction_set_sizes = tf.constant([
            len(p.post) for p in properties
        ])
        disjunction_set_offsets = tf.cumsum(
            disjunction_set_sizes, exclusive=True)

        self._heuristic_order = (
            tf.RaggedTensor.from_row_lengths(
                tf.range(tf.reduce_sum(disjunction_set_sizes)), 
                disjunction_set_sizes) - disjunction_set_offsets[:,None]
        ).merge_dims(0,1)[None]

    def __call__(self, y):
        return tf.zeros_like(y[:,0,None], dtype='int32') + self._heuristic_order
