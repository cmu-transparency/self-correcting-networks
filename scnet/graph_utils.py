import numpy as np
import tensorflow as tf

from functools import partial


class infix(object):
    def __init__(self, func):
        self.func = func

    def __or__(self, other):
        return self.func(other)

    def __ror__(self, other):
        return infix(partial(self.func, other))

    def __call__(self, v1, v2):
        return self.func(v1, v2)


def graph_union(graphs):
    # Assumes the graphs have shape
    #   (batch_size, num_graphs, num_vertices, num_vertices)
    return tf.clip_by_value(tf.reduce_sum(graphs, axis=1), 0, 1)


def contains_cycle_slow(G):
  
    def cond(i, G_next, cycle):
        return i < G.shape[-1]

    def body(i, G_next, cycle):
        return (
            i + 1, 
            G_next @ G, 
            tf.logical_or(cycle, tf.not_equal(tf.linalg.trace(G_next), 0)))

    return tf.while_loop(
        cond, body, [0, G, tf.zeros(G.shape[0], dtype='bool')])[-1]

def contains_cycle(G):
    return tf.linalg.trace(all_pairs_longest_paths_batched(G)) > 0


def distance_product(A1, A2):
    return tf.reduce_min(tf.transpose(A1)[None] + A2[:,None], axis=2)

@infix
def x(A1, A2):
    return distance_product(A1, A2)


def max_distance_product(A1, A2):
    return tf.reduce_max(tf.transpose(A1)[None] + A2[:,None], axis=2)

@infix
def X(A1, A2):
    return max_distance_product(A1, A2)

def max_distance_product_batched(A1, A2):
    return tf.reduce_max(
        tf.transpose(A1, (0,2,1))[:,None] + A2[:,:,None], axis=3)


def set_index(x, i, val):
    mask = tf.eye(x.shape[0], dtype=x.dtype)[i]

    return x * (1 - mask) + val * mask


def all_pairs_shortest_paths(G):
    G = tf.where(
        tf.cast(G, 'bool'), 
        1., 
        tf.where(tf.eye(G.shape[0], dtype='bool'), 0., np.inf))

    def cond(i, paths):
        return i < G.shape[-1]

    def body(i, paths):
        return 2 * i, paths |x| paths

    return tf.while_loop(cond, body, [1, G])[-1]


def all_pairs_longest_paths(G):
    G = tf.where(
        tf.cast(G, 'bool'), 
        1., 
        tf.where(tf.eye(G.shape[0], dtype='bool'), 0., -np.inf))

    def cond(i, paths):
        return i < G.shape[-1]

    def body(i, paths):
        return 2 * i, paths |X| paths

    return tf.while_loop(cond, body, [1, G])[-1]

def all_pairs_longest_paths_batched(G):
    G = tf.where(
        tf.cast(G, 'bool'), 
        1., 
        tf.where(tf.eye(G.shape[-1], dtype='bool')[None], 0., -np.inf))

    def cond(i, paths):
        return i < G.shape[-1]

    def body(i, paths):
        return 2 * i, max_distance_product_batched(paths, paths)

    return tf.while_loop(cond, body, [1, G])[-1]


def get_roots_batched(G, dtype='float32'):
    return tf.cast(
        tf.equal(tf.reduce_max(all_pairs_longest_paths_batched(G), axis=1), 0), 
        dtype)


def topological_sort_semistable(y, partial_order):
    y_norm = y - tf.reduce_min(y)
    y_norm = y_norm / (tf.reduce_max(y_norm) + 1e-5)
    
    return tf.gather(
        tf.sort(y, direction='DESCENDING'),
        tf.math.invert_permutation(tf.argsort(
            tf.reduce_max(all_pairs_longest_paths(partial_order), axis=0) -
                y_norm)))

def topological_sort_stable(y, partial_order):
    return topological_sort((y, partial_order))

def topological_sort(y_partial_order):
    y, partial_order = y_partial_order

    argsort_y = tf.argsort(y, direction='DESCENDING')

    y_sorted = tf.gather(y, argsort_y)

    # Convert y to a total ordering. This makes it so we can decide how to break
    # ties later by adding a constant between 0 and 1.
    y_ord = tf.cast(
        y.shape[-1] - tf.math.invert_permutation(argsort_y),
        'float32')

    longest_paths = all_pairs_longest_paths(partial_order)

    # Whenever y_i should be greater than y_j according to the partial order,
    # but we find that y_j > y_i, set y_j := y_i.
    #
    # This gives us that forall i,j, if (y_i, y_j) is in the edge set of the
    # partial order, y_i >= y_j.
    #
    # Note that the predicted class will always be a source in the dependency
    # graph. This means that the predicted class will be preserved when it is
    # safe w.r.t. our safety properties.
    equal_where_dependent = tf.reduce_min(
        y_ord[:,None] * tf.where(longest_paths >= 0, 1., np.inf), 
        axis=0)

    # Now we have that when y_i is supposed to be greater than y_j, we have that
    # y_i >= y_j. We need to break the ties to get strict inequality. We do this
    # by breaking ties by depth in the partial order DAG. This works because no
    # vertex can have a smaller depth than its parent.
    depths = tf.reduce_max(longest_paths, axis=0)
    max_depth = tf.reduce_max(depths)
    depth_score = (max_depth - depths) / (max_depth + 1.)

    # We can still have ties between vertices at the same depth with the same
    # parent (or child if preserving 'TAIL'). Break ties according to the
    # original ordering.
    tie_breaker_score = y_ord / (y.shape[-1])**2

    new_order = tf.math.invert_permutation(
        tf.argsort(
            equal_where_dependent + depth_score +tie_breaker_score, 
            direction='DESCENDING'))

    return tf.gather(y_sorted, new_order)


def partial_order_satisfied(y, partial_order):
    return tf.reduce_all(
        y[:,:,None] > tf.where(
            tf.cast(partial_order, 'bool'), 
            y[:,None] * tf.cast(partial_order, 'float32'), 
            -np.inf),
        axis=(1,2))
