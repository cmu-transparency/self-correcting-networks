import tensorflow as tf

from scnet.utils import set_index


def solve_disjunction(
    check_sat, 
    required_properties, 
    set_sizes,
    disjunction_set_offsets,
    heuristic_order=None,
    disjunction_set_sizes=None,
    effort=(1<<31) - 1,
):
    max_depth = set_sizes.shape[1]

    # Shape: (None,)
    max_effort = tf.minimum(tf.reduce_prod(set_sizes, axis=1), effort)
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

    if heuristic_order is None:
        if disjunction_set_sizes is None:
            raise ValueError(
                'must specify `disjunction_set_sizes` if `heuristic_order` is '
                'None')
        heuristic_order = tf.zeros_like(set_sizes[:,0,None]) + (
            tf.RaggedTensor.from_row_lengths(
                tf.range(tf.reduce_sum(disjunction_set_sizes)), 
                disjunction_set_sizes) - disjunction_set_offsets[:,None]
        ).merge_dims(0,1)[None]

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
        heuristic_order,
    ):
        # The next index in order is `min_i`. In order to follow the priority
        # speciified by our heuristic, we use this as an index into the index
        # priority list.
        choices = tf.gather(heuristic_order, min_i + disjunction_set_offsets)
        return tf.logical_and(
            tf.logical_not(check_sat(
                choices[None], required_properties[None])[0]),
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
        heuristic_order,
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
            required_properties,
            heuristic_order)

    def next_i(inputs):
        set_sizes, required_properties, heuristic_order = inputs

        max_effort = tf.minimum(tf.reduce_prod(set_sizes), effort)
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

        _, _, _, _, choices, _, _, _, _, _, _ = tf.while_loop(
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
                heuristic_order,
            ])

        return choices

    choices = tf.vectorized_map(
        next_i, 
        (set_sizes, required_properties, heuristic_order))

    # Map the solutions back to the original order.
    return tf.gather(heuristic_order, choices, batch_dims=1)
