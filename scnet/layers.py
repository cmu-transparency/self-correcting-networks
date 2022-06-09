import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer

from scnet.graph_utils import contains_cycle
from scnet.graph_utils import graph_union
from scnet.graph_utils import partial_order_satisfied
from scnet.graph_utils import topological_sort
from scnet.heuristics import Heuristic
from scnet.heuristics import NoHeuristic
from scnet.heuristics import PreservePrediction
from scnet.heuristics import SoftPreservePrediction
from scnet.sat_solving import solve_disjunction
from scnet.properties import MinMaxOrdering
from scnet.utils import AllCombinationsIterator


class MinMaxOrderingRepair(Layer):
    def __init__(self, properties, check_only=False, **kwargs):
        super().__init__(**kwargs)

        self._check_only = check_only

        self._properties = properties

        self._preconditions = [
            p.pre for p in properties
        ]

        self._postconditions = [
            p.post for p in properties
        ]

        self._disjunction_set_sizes = tf.constant([
            len(p.post) for p in properties
        ])
        self._disjunction_set_offsets = tf.cumsum(
            self._disjunction_set_sizes, exclusive=True)

        self._postcondition_mask = tf.concat(
            [
                post_i.mask 
                for p in properties 
                    for post_i in p.post
            ],
            axis=0)

    @property
    def properties(self):
        return self._properties

    @property
    def preconditions(self):
        return self._preconditions

    @property
    def postconditions(self):
        return self._postconditions

    @property
    def postcondition_mask(self):
        return self._postcondition_mask

    def call(self, x_y):
        x, y = x_y

        required_properties = tf.cast(
            tf.concat(
                [pre.sat(x)[:,None] for pre in self.preconditions], axis=1),
            'float32')

        disjunction_choices = tf.where(
            tf.cast(required_properties, 'bool'),
            self._disjunction_set_sizes[None],
            1)

        def sat_check(disjunction_selections, required_properties):
            property_mask = tf.reduce_sum(
                required_properties[:,:,None] *
                    tf.one_hot(
                        disjunction_selections + 
                            self._disjunction_set_offsets[None],
                        self.postcondition_mask.shape[0]),
                axis=1)

            mask = tf.clip_by_value(
                tf.reduce_sum(
                    property_mask[:,:,None,None] * 
                        self.postcondition_mask[None],
                    axis=1),
                0.,
                1.)

            return MinMaxOrdering.satisfiable(mask)

        disjunction_selections = AllCombinationsIterator(
            sat_check, required_properties).next(disjunction_choices)

        property_mask = tf.reduce_sum(
            required_properties[:,:,None] *
                tf.one_hot(
                    disjunction_selections + 
                        self._disjunction_set_offsets[None],
                    self.postcondition_mask.shape[0]),
            axis=1)

        postcondition = MinMaxOrdering(tf.clip_by_value(
            tf.reduce_sum(
                property_mask[:,:,None,None] * self.postcondition_mask[None],
                axis=1),
            0.,
            1.))

        if not self._check_only:
            y = postcondition.fix(y)

        # Check if the new `y` satisfies the postcondition, and if not, set 
        # \bot as the predicted class.
        bot_col = tf.where(postcondition.check(y), -np.inf, np.inf)[:,None]

        return tf.concat([y, bot_col], axis=1)


class GeneralOrderingRepair(Layer):
    def __init__(
        self, properties, heuristic='preserve', check_only=False, **kwargs
    ):
        super().__init__(**kwargs)

        self._properties = properties
        self._check_only = check_only

        self._preconditions = [
            p.pre for p in properties
        ]

        self._postconditions = [
            p.post for p in properties
        ]

        self._disjunction_set_sizes = tf.constant([
            len(p.post) for p in properties
        ])
        self._disjunction_set_offsets = tf.cumsum(
            self._disjunction_set_sizes, exclusive=True)

        if isinstance(heuristic, Heuristic):
            self._heuristic = heuristic

        elif heuristic == 'preserve':
            self._heuristic = PreservePrediction()

        elif heuristic == 'soft_preserve':
            self._heuristic = SoftPreservePrediction()

        elif heuristic is None or heuristic == 'none':
            self._heuristic = NoHeuristic()

        else:
            raise ValueError(f'unknown heuristic: {heuristic}')

    @property
    def properties(self):
        return self._properties

    @property
    def preconditions(self):
        return self._preconditions

    @property
    def postconditions(self):
        return self._postconditions

    @property
    def postcondition_graph(self):
        return self._postcondition_graph

    def build(self, input_shape):
        x_shape, y_shape = input_shape

        num_classes = y_shape[-1]

        self._postcondition_graph = tf.concat(
            [
                p.post.make_graph(num_classes) 
                for p in self.properties
            ],
            axis=0)

        self._heuristic.build(self.properties, num_classes)

    def call(self, x_y):
        x, y = x_y

        required_properties = tf.cast(
            tf.concat(
                [pre.sat(x, y)[:,None] for pre in self.preconditions], axis=1),
            'int32')

        init_disjunction_choices = tf.where(
            tf.cast(required_properties, 'bool'),
            self._disjunction_set_sizes[None],
            1)

        def sat_check(disjunction_selections, required_properties):
            property_mask = tf.reduce_sum(
                required_properties[:,:,None] *
                    tf.one_hot(
                        disjunction_selections + 
                            self._disjunction_set_offsets[None],
                        self.postcondition_graph.shape[0],
                        dtype='int32'),
                axis=1)

            graph = graph_union(
                property_mask[:,:,None,None] * self.postcondition_graph[None])

            return tf.logical_not(contains_cycle(graph))
          
        disjunction_selections = solve_disjunction(
            sat_check, 
            required_properties, 
            init_disjunction_choices,
            self._disjunction_set_offsets,
            disjunction_set_sizes=self._disjunction_set_sizes,
            heuristic_order=self._heuristic(y))

        property_mask = tf.reduce_sum(
            required_properties[:,:,None] *
                tf.one_hot(
                    disjunction_selections + 
                        self._disjunction_set_offsets[None],
                    self.postcondition_graph.shape[0],
                    dtype='int32'),
            axis=1)

        postcondition_graph = graph_union(
            property_mask[:,:,None,None] * self.postcondition_graph[None])

        y_fixed = tf.vectorized_map(
            topological_sort, 
            (y, postcondition_graph))
        
        # If we already satisfy the properties, we should return `y`, since we 
        # might not end up picking the right disjunction that would lead to the
        # repaired output being the same as the original output.
        if not self._check_only:
            y = tf.where(
                partial_order_satisfied(y, postcondition_graph)[:,None], 
                y, 
                y_fixed)
        
        # Check if the new `y` satisfies the postcondition, and if not, set 
        # \bot as the predicted class.
        margin = tf.reduce_sum(
            tf.nn.relu(
                -y[:,:,None] + tf.where(
                    tf.cast(postcondition_graph, 'bool'), 
                    y[:,None] * tf.cast(postcondition_graph, 'float32'), 
                    -np.inf)), 
            axis=(1,2))

        bot_col = tf.where(
            margin > 0., 
            tf.reduce_max(y, axis=1) + margin,
            -np.inf)[:,None]
        
        return tf.concat([y, bot_col], axis=1)
