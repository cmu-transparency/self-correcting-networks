import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Progbar

from scnet.graph_utils import all_pairs_longest_paths
from scnet.graph_utils import contains_cycle
from scnet.graph_utils import graph_union
from scnet.order_properties import ComparisonConjunction
from scnet.order_properties import ComparisonDisjunction
from scnet.order_properties import OrderingPostcondition
from scnet.order_properties import Output
from scnet.properties import Property
from scnet.properties import InputInRange
from scnet.utils import BatchedAllCombinations


def random_precondition(input_dimensions, r=0.33):
    lower = np.random.rand(input_dimensions) * (1. - r)

    return InputInRange(lower, lower + r)


def random_conjunction(num_classes, expected_comps=2):
    n = num_classes**2 - num_classes
    num_comps = np.random.binomial(n=n, p=(expected_comps - 1.) / n) + 1

    has_cycle = True
    while has_cycle:
        # Avoid the diagonal, since we know that will always create a cycle.
        permutation = np.random.permutation(num_classes**2)
        permutation = permutation[permutation % num_classes != 0]

        comp_index = permutation[:num_comps]

        G = np.zeros(num_classes**2)
        G[comp_index] = 1.
        G = G.reshape(num_classes, num_classes)
        
        has_cycle = contains_cycle(G[None])[0]
    
    # Convert `G` to an `OrderingPostcondition`.
    return ComparisonConjunction(*[
        Output(i) >> Output(j)
        for i in range(num_classes)
            for j in range(num_classes)
        if G[i][j] == 1.
    ])

def random_postcondition(num_classes, disjuncts=1, expected_comps=2):
    return OrderingPostcondition(ComparisonDisjunction(*[
        random_conjunction(num_classes, expected_comps=expected_comps)
        for _ in range(disjuncts)
    ]))


def random_property(
    input_dimensions, num_classes, r=0.33, disjuncts=1, expected_comps=2
):
    return Property(
        random_precondition(input_dimensions, r),
        random_postcondition(num_classes, disjuncts, expected_comps))


def points_matching_preconditions(properties, N):
    return np.concatenate(
        [
            p.pre.lower[None] + 
                (p.pre.upper - p.pre.lower) *
                np.random.rand(N, *p.pre.lower.shape) 

            for p in properties
        ],
        axis=0)

def points_not_matching_preconditions(properties, N):
    X = []
    total = 0
    while total < N:
        x = np.random.rand(N, *properties[0].pre.lower.shape)

        where_no_match = tf.logical_not(
            tf.reduce_any(
                tf.concat(
                    [
                        p.pre.sat(x)[:,None]
                        for p in properties
                    ], 
                    axis=1), 
                axis=1)).numpy()
        
        X.append(x[where_no_match])
        
        total += where_no_match.sum()
        
    return np.concatenate(X, axis=0)[:N]


def label_points(X, properties, num_classes, effort=5):
    N = len(X)
    
    # Get the properties satisfied on each point
    required_properties = tf.cast(
      tf.concat(
          [p.pre.sat(X)[:,None] for p in properties], axis=1),
      'int32')
  
    
    # List the possible choices for selecting a disjunct from each property.
    disjunction_set_sizes = tf.constant([
        len(p.post) for p in properties
    ])
    disjunction_set_offsets = tf.cumsum(
        disjunction_set_sizes, exclusive=True)
    
    init_disjunction_choices = tf.where(
        tf.cast(required_properties, 'bool'),
        disjunction_set_sizes[None],
        1)
    
    disjunction_choices = BatchedAllCombinations(effort=effort).next(
        init_disjunction_choices)
    
    # Figure out which choices are satisfiable
    disjunction_selections = tf.reshape(disjunction_choices, (N*effort, -1))
    required_properties = tf.repeat(required_properties, effort, axis=0)

    postcondition_graph = tf.concat(
        [
            p.post.make_graph(num_classes) 
            for p in properties
        ],
        axis=0)
    
    property_mask = tf.reduce_sum(
        required_properties[:,:,None] *
            tf.one_hot(
                disjunction_selections + 
                    disjunction_set_offsets[None],
                postcondition_graph.shape[0],
                dtype='int32'),
        axis=1)
    
    postcondition_graph = graph_union(
        property_mask[:,:,None,None] * postcondition_graph[None])
    
    
    sat = tf.logical_not(contains_cycle(postcondition_graph))
    
    # For each input, give a label that is allowed by the ordering constraint.
    sat = tf.reshape(sat, (N, effort)).numpy()
    postcondition_graph = tf.reshape(
        postcondition_graph, (N, effort, num_classes, num_classes))
    
    new_X, y = [], []
    for i in range(N):
        if not np.any(sat[i]):
            # Skip inputs that are unsatisfiable.
            print('found unsatisfiable point!')
            continue
            
        G = postcondition_graph[i][sat[i]][np.random.choice(sat[i].sum())]
        
        possible_classes = np.arange(num_classes)[
            tf.reduce_max(all_pairs_longest_paths(G), axis=0).numpy() == 0
        ]
        label = possible_classes[np.random.choice(len(possible_classes))]
        
        new_X.append(X[i])
        y.append(label)
    
    return np.array(new_X), np.array(y)


class SyntheticOrderingPropertyData(object):

    def __init__(
        self, 
        input_dimensions, 
        num_classes, 
        num_properties,
        size,
        fraction_in_properties=0.5,
        r=0.33, 
        disjuncts=1, 
        expected_comps=2,
        effort=5,
        creation_batch_size=10,
        verbose=False,
    ):
        self._input_shape = (input_dimensions,)
        self._num_classes = num_classes
        self._fraction_in_properties = fraction_in_properties
        self._r = r
        self._disjuncts = disjuncts
        self._expected_comps = expected_comps
        self._effort = effort

        self._properties = [
            random_property(
                input_dimensions, num_classes, 
                r=r, 
                disjuncts=disjuncts, 
                expected_comps=expected_comps)
            for _ in range(num_properties)
        ]

        if verbose:
            print('making points in preconditions...')
        X, Y = SyntheticOrderingPropertyData.make(
            2 * size, 
            self.properties, 
            num_classes, 
            effort=effort, 
            batch_size=creation_batch_size, 
            verbose=verbose)

        self._x_tr = X[:len(X) // 2]
        self._x_te = X[len(X) // 2:]
        self._y_tr = Y[:len(Y) // 2]
        self._y_te = Y[len(Y) // 2:]
        self._y_tr_1hot = tf.one_hot(self.y_tr, num_classes)
        self._y_te_1hot = tf.one_hot(self.y_te, num_classes)

    @property
    def x_tr(self):
        return self._x_tr

    @property
    def y_tr(self):
        return self._y_tr

    @property
    def y_tr_1hot(self):
        return self._y_tr_1hot

    @property
    def x_te(self):
        return self._x_te

    @property
    def y_te(self):
        return self._y_te

    @property
    def y_te_1hot(self):
        return self._y_te_1hot

    @property
    def properties(self):
        return self._properties

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def fraction_in_properties(self):
        return self._fraction_in_properties

    @property
    def r(self):
        return self._r

    @property
    def disjuncts(self):
        return self._disjuncts

    @property
    def expected_comps(self):
        return self._expected_comps

    def save(self, file_name):
        with open(file_name + '.synth', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name):
        with open(file_name + '.synth', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def make(
        N, 
        properties, 
        num_classes, 
        fraction_in_properties=0.5,
        effort=5, 
        batch_size=10, 
        shuffle=True,
        verbose=False,
    ):
        if verbose:
            pb = Progbar(N // batch_size)
            pb.add(0)

        X, Y = [], []
        for _ in range(N // batch_size):
            x, y = label_points(
                points_matching_preconditions(properties, batch_size), 
                properties, 
                num_classes, 
                effort=effort)

            X.append(x)
            Y.append(y)

            if verbose:
                pb.add(1)

        # Make points that aren't captured by the properties.
        if verbose:
            print('adding points not covered by preconditions')

        M = int(
            N * len(properties) / fraction_in_properties - N * len(properties))

        x = points_not_matching_preconditions(properties, M)

        # Label the points using a model that fits the previous data.
        input_shape = properties[0].pre.lower.shape

        f = Sequential()
        f.add(Dense(100, activation='relu', input_shape=input_shape))
        f.add(Dense(100, activation='relu'))
        f.add(Dense(100, activation='relu'))
        f.add(Dense(num_classes, activation='softmax'))

        f.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics='acc')

        f.fit(
            np.concatenate(X, axis=0),
            tf.one_hot(np.concatenate(Y, axis=0), num_classes),
            epochs=100,
            batch_size=batch_size,
            verbose=1 if verbose else 0)

        y = f.predict(x).argmax(axis=1)

        # Put everything together.
        X.append(x)
        Y.append(y)
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        # Shuffle if necessary.
        if shuffle:
            order = np.random.permutation(len(X))

            X = X[order]
            Y = Y[order]

        return X, Y


if __name__ == '__main__':

    import os

    from scriptify import scriptify

    @scriptify
    def main(
        name,
        input_dimensions, 
        num_classes, 
        num_properties,
        total_size,
        fraction_in_properties=0.5,
        precond_radius=0.33, 
        disjuncts=1, 
        expected_comparisons=2,
        effort=5,
        creation_batch_size=10,
        no_overwrite=False,
    ):
        if 'SYNTHETIC_DATA_DIR' in os.environ:
            path = os.environ['SYNTHETIC_DATA_DIR']

            if not path.endswith('/'):
                path += '/'
        else:
            raise RuntimeError(
                'please set the "SYNTHETIC_DATA_DIR" environment variable')

        size = int(total_size * fraction_in_properties / num_properties)

        data = SyntheticOrderingPropertyData(
            input_dimensions,
            num_classes,
            num_properties,
            size,
            fraction_in_properties=fraction_in_properties,
            r=precond_radius,
            disjuncts=disjuncts,
            expected_comps=expected_comparisons,
            effort=effort,
            creation_batch_size=creation_batch_size,
            verbose=True)

        if no_overwrite and os.path.isfile(path + name + '.synth'):
            raise ValueError(f'file {path + name}.synth already exists')

        data.save(path + name)

        return {
            'location': path + name
        }
