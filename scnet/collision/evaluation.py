import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scriptify import scriptify
from tensorflow.data import Dataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import Progbar
from time import time

from scnet.layers import GeneralOrderingRepair
from scnet.properties import Property
from scnet.properties import InputInRange
from scnet.order_properties import OrderingPostcondition
from scnet.order_properties import Output


def get_collision_model():
    return load_model('models/original.h5')

def get_collision_data(batch_size=128):
    test = np.load('data/test.npz')

    # Convert to tensorflow dataset.
    test = (Dataset.from_tensor_slices((test['x'], test['y']))
        .batch(batch_size))

    return test

def get_collision_properties():
    npz = np.load('data/properties.npz')
    
    return [
        Property(
            InputInRange(lb, ub),
            OrderingPostcondition(Output(order[0]) < Output(order[1])))
        for lb, ub, order in zip(npz['lbs'], npz['ubs'], npz['orders'])
    ]


def fraction_in_violation(y_true, y_pred):
  return tf.reduce_mean(
    tf.cast(
        tf.equal(tf.argmax(y_pred, axis=1), y_pred.shape[-1] - 1), 'float32'))


if __name__ == '__main__':

    @scriptify
    def main(batch_size=128, heuristic='none', gpu=0):

        # Select the GPU and allow memory growth to avoid taking all the RAM.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        device = gpus[gpu]

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        input_shape = (6,)
        num_classes = 2
        num_test_points = 3000


        test = get_collision_data(batch_size=batch_size)

        properties = get_collision_properties()

        f = get_collision_model()

        x = f.input
        y = f.output

        f_sr = Model(x, GeneralOrderingRepair(properties, heuristic)([x,y]))

        f.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(),
            metrics=['acc'])
        f_sr.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(),
            metrics=['acc', fraction_in_violation])

        start_time = time()

        _, acc_orig = f.evaluate(test)

        time_per_instance_orig = (time() - start_time) / num_test_points

        f_sr.summary()

        # Dry run first to "warm up," since this model hasn't been run yet.
        f_sr.evaluate(test, verbose=False)

        start_time = time()

        _, acc_sr, violations = f_sr.evaluate(test)

        time_per_instance_sr = (time() - start_time) / num_test_points

        assert violations == 0., f'violations: {violations}'

        print('-'*80)
        print(
            f'\t| accuracy\t| time\n'
            f'orig\t| {acc_orig:.4f}\t| {time_per_instance_orig:.5f}\n'
            f'SR\t| {acc_sr:.4f}\t| {time_per_instance_sr:.5f}\n')

        return {
            'acc_orig': float(np.mean(acc_orig)),
            'acc_sr': float(np.mean(acc_sr)),
            'time_orig': float(np.mean(time_per_instance_orig)),
            'time_sr': float(np.mean(time_per_instance_sr)),
        }
