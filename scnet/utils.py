import numpy as np
import tensorflow as tf

from functools import partial


class Infix(object):
    def __init__(self, func):
        self.func = func
    def __or__(self, other):
        return self.func(other)
    def __ror__(self, other):
        return Infix(partial(self.func, other))
    def __call__(self, v1, v2):
        return self.func(v1, v2)


def rotate_row(direction):
    if direction == 'from_top':
        direction = 'DESCENDING'

    elif direction == 'from_bottom':
        direction = 'ASCENDING'

    def rotate_row(amount_concat_with_y):
        amount = tf.cast(amount_concat_with_y[0], 'int32')
        y = amount_concat_with_y[1:]

        y_sorted = tf.sort(y, direction=direction)
        y_perm = tf.argsort(y, direction=direction)

        y_perm_inverse = tf.math.invert_permutation(y_perm)

        y_front = y_sorted[:amount+1]
        y_back = y_sorted[amount+1:]

        y_roll = tf.roll(y_front, amount, axis=0)

        y_final = tf.concat([y_roll, y_back], axis=0)

        return tf.gather(y_final, y_perm_inverse)

    return rotate_row


def contains_cycle(E):
  
    def cond(i, E_next, cycle):
        return i < E.shape[-1]

    def body(i, E_next, cycle):
        return (
            i + 1, 
            E_next @ E, 
            tf.logical_or(cycle, tf.not_equal(tf.linalg.trace(E_next), 0)))

    return tf.while_loop(cond, body, [0, E, False])[-1]


def distance_product(A1, A2):
    return tf.reduce_min(tf.transpose(A1)[None] + A2[:,None], axis=2)

@Infix
def x(A1, A2):
    return distance_product(A1, A2)


def max_distance_product(A1, A2):
    return tf.reduce_max(tf.transpose(A1)[None] + A2[:,None], axis=2)

@Infix
def X(A1, A2):
    return max_distance_product(A1, A2)


def set_index(x, i, val):
    mask = tf.eye(x.shape[0], dtype=x.dtype)[i]

    return x * (1 - mask) + val * mask


def all_pairs_shortest_paths(E):
    E = tf.where(
        tf.cast(E, 'bool'), 
        1., 
        tf.where(tf.eye(E.shape[0], dtype='bool'), 0., -np.inf))

    def cond(i, paths):
        return i < E.shape[-1]

    def body(i, paths):
        return 2 * i, paths |x| paths

    return tf.while_loop(cond, body, [1, E])[-1]


def all_pairs_longest_paths(E):
    E = tf.where(
        tf.cast(E, 'bool'), 
        1., 
        tf.where(tf.eye(E.shape[0], dtype='bool'), 0., -np.inf))

    def cond(i, paths):
        return i < E.shape[-1]

    def body(i, paths):
        return 2 * i, paths |X| paths

    return tf.while_loop(cond, body, [1, E])[-1]


def topological_sort(y, partial_order):
    y_norm = y - tf.reduce_min(y)
    y_norm = y / (tf.reduce_max(y) + 1e-5)
    
    return tf.gather(
        tf.sort(y, direction='DESCENDING'),
        tf.math.invert_permutation(tf.argsort(
            tf.reduce_max(all_pairs_longest_paths(partial_order), axis=0) -
                y_norm)))


class AllCombinationsIteratorOld(object):

    def __init__(self, set_sizes):
        self._set_sizes = tf.constant(set_sizes)

        self._len = tf.reduce_prod(set_sizes)
        self._num_sets = len(set_sizes)

        self._max_sum = tf.reduce_sum(set_sizes)
        self._max_sums = tf.cumsum(
            np.array(set_sizes[::-1], dtype='int32') - 1, 
            exclusive=True)[::-1]

        self.__sum = tf.Variable(-1, trainable=False)
        self.__depth = tf.Variable(0, trainable=False)
        self.__cumsum = tf.Variable(0, trainable=False)
        self.__min_i = tf.Variable([0 for _ in self._set_sizes])
        self.__max_i = tf.Variable([0 for _ in self._set_sizes])

    def reset(self):
        self.__sum.assign(-1)
        self.__depth.assign(0)
        self.__cumsum.assign(0)
        self.__min_i.assign(0 * self.__min_i)
        self.__max_i.assign(0 * self.__min_i)

        return self

    def next(self):

        # Pop back to next option.
        def pop_cond(depth, _):
            return tf.logical_and(
                depth >= 0, self.__min_i[depth] == self.__max_i[depth])

        def pop_body(depth, cumsum):
            return depth - 1, cumsum - self.__min_i[depth]

        depth, cumsum = tf.while_loop(
            pop_cond, pop_body, [self.__depth, self.__cumsum])

        self.__depth.assign(depth)
        self.__cumsum.assign(cumsum)

        # Check if we need to update the sum.
        update_sum = tf.cast(self.__depth < 0, 'int32')

        self.__sum.assign_add(update_sum)

        # Otherwise, advance the min index.
        self.__min_i[self.__depth].assign(
            self.__min_i[self.__depth] + (1 - update_sum))
        self.__cumsum.assign_add(1 - update_sum)

        # Reset the ranges on the next options.
        def reset_range_cond(i, min_i, max_i, depth, cumsum):
            return i < self._num_sets

        def reset_range_body(i, min_i, max_i, depth, cumsum):
            new_min_i = tf.maximum(0, self.__sum - cumsum - self._max_sums[i])
            new_max_i = tf.minimum(self._set_sizes[i] - 1, self.__sum - cumsum)
            return (
                i + 1,
                set_index(min_i, i, new_min_i),
                set_index(max_i, i, new_max_i),
                depth + 1,
                cumsum + new_min_i)

        _, min_i, max_i, depth, cumsum = tf.while_loop(
            reset_range_cond, 
            reset_range_body, 
            [
                self.__depth + 1, 
                self.__min_i, 
                self.__max_i, 
                self.__depth, 
                self.__cumsum,
            ])

        self.__min_i.assign(min_i)
        self.__max_i.assign(max_i)
        self.__depth.assign(depth)
        self.__cumsum.assign(cumsum)

        return min_i


class AllCombinationsIterator(object):

    def __init__(self, check_sat, required_properties, effort=None):

        if effort is None:
            effort = 2**31 - 1

        self._check_sat = check_sat
        self._required_properties = required_properties
        self._effort = effort

    def next(self, set_sizes):

        max_depth = set_sizes.shape[1]

        # Shape: (None,)
        max_effort = tf.minimum(
            tf.reduce_prod(set_sizes, axis=1),
            self._effort)
        max_sum = tf.reduce_sum(set_sizes - 1, axis=1)

        # Shape: (None, max_depth)
        max_partial_sum = tf.cumsum(
            set_sizes[:,::-1] - 1,
            axis=1,
            exclusive=True)[:,::-1]

        # Shape: (None,)
        total = tf.zeros_like(max_sum) - 1
        depth = tf.zeros_like(max_sum)
        cumsum = tf.zeros_like(max_sum)

        # Shape: (None, max_depth)
        min_i = tf.zeros_like(set_sizes)
        max_i = tf.zeros_like(set_sizes)

        # Pop back to next option.
        def pop_cond(total, depth, cumsum, min_i, max_i):
            return tf.logical_and(
                depth >= 0, min_i[depth] == max_i[depth])

        def pop_body(total, depth, cumsum, min_i, max_i):
            return (
                total,
                depth - 1, 
                cumsum - min_i[depth],
                min_i,
                max_i)

        # Reset the ranges on the next options.
        def reset_range_cond(
            i, total, depth, cumsum, min_i, max_i, max_partial_sum, set_sizes
        ):
            return i < max_depth

        def reset_range_body(
            i, total, depth, cumsum, min_i, max_i, max_partial_sum, set_sizes
        ):
            new_min_i = tf.maximum(0, total - cumsum - max_partial_sum[i])
            new_max_i = tf.minimum(set_sizes[i] - 1, total - cumsum)
            return (
                i + 1,
                total,
                depth + 1,
                cumsum + new_min_i,
                set_index(min_i, i, new_min_i),
                set_index(max_i, i, new_max_i),
                max_partial_sum,
                set_sizes)

        def next_cond(
            j, 
            total, 
            depth, 
            cumsum, 
            min_i, 
            max_i, 
            max_partial_sum, 
            set_sizes, 
            max_effort,
            required_properties,
        ):
            return tf.logical_and(
                tf.logical_not(self._check_sat(
                    min_i[None], required_properties[None])[0]),
                j < max_effort)

        def next_body(
            j, 
            total, 
            depth, 
            cumsum, 
            min_i, 
            max_i, 
            max_partial_sum, 
            set_sizes, 
            max_effort,
            required_properties,
        ):
            _, depth, cumsum, _, _ = tf.while_loop(
                pop_cond, pop_body, [total, depth, cumsum, min_i, max_i])
            
            # Check if we need to update the sum.
            update_sum = tf.cast(depth < 0, 'int32')

            total += update_sum

            # Otherwise, advance the min index.
            min_i = set_index(
                min_i, depth, min_i[depth] + (1 - update_sum))
            cumsum += 1 - update_sum

            _, _, depth, cumsum, min_i, max_i, _, _ = tf.while_loop(
            reset_range_cond, 
            reset_range_body, 
            [
                depth + 1, 
                total,
                depth,
                cumsum,
                min_i,
                max_i,
                max_partial_sum,
                set_sizes,
            ])
            
            return (
                j + 1,
                total, 
                depth, 
                cumsum, 
                min_i, 
                max_i,
                max_partial_sum,
                set_sizes,
                max_effort,
                required_properties)

        def next_i(inputs):
            set_sizes, required_properties = inputs

            max_effort = tf.minimum(tf.reduce_prod(set_sizes), self._effort)
            max_sum = tf.reduce_sum(set_sizes - 1)

            max_partial_sum = tf.cumsum(
                set_sizes[::-1] - 1,
                axis=0,
                exclusive=True)[::-1]

            total = tf.zeros_like(max_sum) - 1
            depth = tf.zeros_like(max_sum)
            cumsum = tf.zeros_like(max_sum)

            min_i = tf.zeros_like(set_sizes)
            max_i = tf.zeros_like(set_sizes)

            _, _, _, _, choices, _, _, _, _, _ = tf.while_loop(
                next_cond,
                next_body,
                [
                    tf.constant(0), 
                    total, 
                    depth, 
                    cumsum, 
                    min_i, 
                    max_i, 
                    max_partial_sum,
                    set_sizes,
                    max_effort,
                    required_properties,
                ])

            return choices

        return tf.map_fn(
            next_i, 
            (set_sizes, self._required_properties), 
            fn_output_signature='int32')


class BatchedAllCombinations(object):

    def __init__(self, effort):

        self._effort = tf.Variable(effort, name='effort', trainable=False)

    def next(self, set_sizes):

        max_depth = set_sizes.shape[1]

        # Shape: (None,)
        max_sum = tf.reduce_sum(set_sizes - 1, axis=1)

        # Shape: (None, max_depth)
        max_partial_sum = tf.cumsum(
            set_sizes[:,::-1] - 1,
            axis=1,
            exclusive=True)[:,::-1]

        # Shape: (None,)
        total = tf.zeros_like(max_sum) - 1
        depth = tf.zeros_like(max_sum)
        cumsum = tf.zeros_like(max_sum)

        # Shape: (None, max_depth)
        min_i = tf.zeros_like(set_sizes)
        max_i = tf.zeros_like(set_sizes)

        # Pop back to next option.
        def pop_cond(total, depth, cumsum, min_i, max_i):
            return tf.logical_and(
                depth >= 0, min_i[depth] == max_i[depth])

        def pop_body(total, depth, cumsum, min_i, max_i):
            return (
                total,
                depth - 1, 
                cumsum - min_i[depth],
                min_i,
                max_i)

        # Reset the ranges on the next options.
        def reset_range_cond(
            i, total, depth, cumsum, min_i, max_i, max_partial_sum, set_sizes
        ):
            return i < max_depth

        def reset_range_body(
            i, total, depth, cumsum, min_i, max_i, max_partial_sum, set_sizes
        ):
            new_min_i = tf.maximum(0, total - cumsum - max_partial_sum[i])
            new_max_i = tf.minimum(set_sizes[i] - 1, total - cumsum)
            return (
                i + 1,
                total,
                depth + 1,
                cumsum + new_min_i,
                set_index(min_i, i, new_min_i),
                set_index(max_i, i, new_max_i),
                max_partial_sum,
                set_sizes)

        def next_i(inputs):
            total, depth, cumsum, min_i, max_i, max_partial_sum, set_sizes = \
                inputs
            
            _, depth, cumsum, _, _ = tf.while_loop(
                pop_cond, pop_body, [total, depth, cumsum, min_i, max_i])
            
            # Check if we need to update the sum.
            update_sum = tf.cast(depth < 0, 'int32')

            total += update_sum

            # Otherwise, advance the min index.
            min_i = set_index(
                min_i, depth, min_i[depth] + (1 - update_sum))
            cumsum += 1 - update_sum

            _, _, depth, cumsum, min_i, max_i, _, _ = tf.while_loop(
            reset_range_cond, 
            reset_range_body, 
            [
                depth + 1, 
                total,
                depth,
                cumsum,
                min_i,
                max_i,
                max_partial_sum,
                set_sizes,
            ])
            
            return total, depth, cumsum, min_i, max_i

        # Main Body.
        def next_cond(
            j, total, depth, cumsum, min_i, max_i, max_partial_sum, output
        ):
            return j < self._effort

        def next_body(
            j, total, depth, cumsum, min_i, max_i, max_partial_sum, output
        ):
            total, depth, cumsum, min_i, max_i = tf.map_fn(
                next_i, 
                (
                    total, 
                    depth, 
                    cumsum, 
                    min_i, 
                    max_i, 
                    max_partial_sum, 
                    set_sizes
                ),
                fn_output_signature=(
                    'int32', 'int32', 'int32', 'int32', 'int32'))
            
            output = tf.concat([output, min_i[:, None]], axis=1)
            
            # Reset if `total` becomes too big.
            restart = tf.cast(total >= max_sum, 'int32')

            def reset(v, initial, cond):
                return v * (1 - cond) + initial * cond

            total = reset(total, tf.zeros_like(max_sum) - 1, restart)
            depth = reset(depth, tf.zeros_like(max_sum), restart)
            cumsum = reset(cumsum, tf.zeros_like(max_sum), restart)
            min_i = reset(min_i, tf.zeros_like(set_sizes), restart[:,None])
            max_i = reset(max_i, tf.zeros_like(set_sizes), restart[:,None])

            return (
                j + 1,
                total,
                depth,
                cumsum,
                min_i,
                max_i,
                max_partial_sum,
                output)

        return tf.while_loop(
            next_cond,
            next_body,
            [
                tf.constant(0), 
                total, 
                depth, 
                cumsum, 
                min_i, 
                max_i, 
                max_partial_sum,
                tf.zeros_like(set_sizes)[:,None][:,0:0]
            ],
            shape_invariants=[
                tf.constant(0).shape, 
                total.shape, 
                depth.shape, 
                cumsum.shape, 
                min_i.shape, 
                max_i.shape, 
                max_partial_sum.shape,
                tf.TensorShape([None, None, set_sizes.shape[-1]])
            ])[-1]
