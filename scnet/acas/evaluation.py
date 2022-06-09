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

from scnet.acas.properties import properties_for_model
from scnet.acas.properties import safe_models
from scnet.acas.properties import unsafe_models
from scnet.layers import GeneralOrderingRepair


def get_acas_model(i, j):
    f = load_model(f'models/ACASXU_run2a_{i+1}_{j+1}_batch_2000.h5')

    # Model does predictions backwards; the predicted class should be the
    # maximal logit, but it is the minimal logit.
    W = f.layers[-1].get_weights()
    W[0] = -W[0]
    W[1] = -W[1]
    f.layers[-1].set_weights(W)
    
    return f

def get_acas_data(i, j, batch_size=128):
    train = np.load(f'data/acas_art.{i+1}.{j+1}.train.npz')
    test = np.load(f'data/acas_art.{i+1}.{j+1}.test.npz')

    # Convert to tensorflow dataset.
    train = (Dataset.from_tensor_slices((train['x'], train['y']))
        .batch(batch_size))
    test = (Dataset.from_tensor_slices((test['x'], test['y']))
        .batch(batch_size))

    return train, test


def fraction_in_violation(y_true, y_pred):
  return tf.reduce_mean(
    tf.cast(
        tf.equal(tf.argmax(y_pred, axis=1), y_pred.shape[-1] - 1), 'float32'))


if __name__ == '__main__':

    @scriptify
    def main(batch_size=128, heuristic='preserve', gpu=0):

        # Select the GPU and allow memory growth to avoid taking all the RAM.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        device = gpus[gpu]

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        input_shape = (5,)
        num_classes = 5
        num_test_points = 5000

        times_safe_orig, accs_safe_orig = [], []
        times_safe_sr, accs_safe_sr = [], []
        for model_id in safe_models:
            print(f'Repairing model {model_id}...')

            _, test = get_acas_data(*model_id, batch_size=batch_size)

            properties = properties_for_model[model_id]

            f = get_acas_model(*model_id)

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

            # Dry run first to "warm up," since this model hasn't been run yet.
            f_sr.evaluate(test, verbose=False)

            start_time = time()

            _, acc_sr, violations = f_sr.evaluate(test)

            time_per_instance_sr = (time() - start_time) / num_test_points

            assert violations == 0., f'violations: {violations}'

            times_safe_orig.append(time_per_instance_orig)
            accs_safe_orig.append(acc_orig)
            times_safe_sr.append(time_per_instance_sr)
            accs_safe_sr.append(acc_sr)

        print('='*80)
        print('9 SAFE MODELS')
        print('-'*80)
        print(
            f'\t| accuracy\t| time\n'
            f'orig\t| {np.mean(accs_safe_orig):.3f}\t\t| '
            f'{np.mean(times_safe_orig):.5f}\n'
            f'SR\t| {np.mean(accs_safe_sr):.3f}\t\t| '
            f'{np.mean(times_safe_sr):.5f}\n')

        times_unsafe_orig, accs_unsafe_orig = [], []
        times_unsafe_sr, accs_unsafe_sr = [], []
        for model_id in unsafe_models:
            print(f'Repairing model {model_id}...')

            _, test = get_acas_data(*model_id, batch_size=batch_size)

            properties = properties_for_model[model_id]

            f = get_acas_model(*model_id)

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

            # Dry run first to "warm up," since this model hasn't been run yet.
            f_sr.evaluate(test, verbose=False)

            start_time = time()

            _, acc_sr, violations = f_sr.evaluate(test)

            time_per_instance_sr = (time() - start_time) / num_test_points

            assert violations == 0., f'violations: {violations}'

            times_unsafe_orig.append(time_per_instance_orig)
            accs_unsafe_orig.append(acc_orig)
            times_unsafe_sr.append(time_per_instance_sr)
            accs_unsafe_sr.append(acc_sr)

        print('='*80)
        print('36 UNSAFE MODELS')
        print('-'*80)
        print(
            f'\t| accuracy\t| time\n'
            f'orig\t| {np.mean(accs_unsafe_orig):.3f}\t\t| '
            f'{np.mean(times_unsafe_orig):.5f}\n'
            f'SR\t| {np.mean(accs_unsafe_sr):.3f}\t\t| '
            f'{np.mean(times_unsafe_sr):.5f}\n')

        return {
            'mean_acc_safe_orig': float(np.mean(accs_safe_orig)),
            'mean_acc_safe_sr': float(np.mean(accs_safe_sr)),
            'time_safe_orig': float(np.mean(times_safe_orig)),
            'time_safe_sr': float(np.mean(times_safe_sr)),
            'mean_acc_unsafe_orig': float(np.mean(accs_unsafe_orig)),
            'mean_acc_unsafe_sr': float(np.mean(accs_unsafe_sr)),
            'time_unsafe_orig': float(np.mean(times_unsafe_orig)),
            'time_unsafe_sr': float(np.mean(times_unsafe_sr)),
        }
