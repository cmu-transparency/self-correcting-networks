import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from gloro.training.callbacks import LrScheduler
from hashlib import blake2b
from scriptify import scriptify
from tensorflow.data import Dataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import Progbar
from time import time

from scnet import architectures
from scnet.layers import GeneralOrderingRepair
from scnet.misc.cifar_properties import get_superclass_properties
from scnet.misc.synthetic_data import SyntheticOrderingPropertyData


def fraction_in_violation(y_true, y_pred):
  return tf.reduce_mean(
    tf.cast(
        tf.equal(tf.argmax(y_pred, axis=1), y_pred.shape[-1] - 1), 'float32'))


def data_augmentation(
    flip=True,
    saturation=(0.5, 1.2),
    contrast=(0.8, 1.2),
    zoom=0.1,
    noise=None
):
    def augment(x, y):
        batch_size = tf.shape(x)[0]
        input_shape = x.shape[1:]

        # Horizontal flips
        if flip:
            x = tf.image.random_flip_left_right(x)

        # Randomly adjust the saturation and contrast.
        if saturation is not None and input_shape[-1] == 3:
            x = tf.image.random_saturation(
                x, lower=saturation[0], upper=saturation[1])

        if contrast is not None:
            x = tf.image.random_contrast(
                x, lower=contrast[0], upper=contrast[1])

        # Randomly zoom.
        if zoom is not None:
            widths = tf.random.uniform([batch_size], 1. - zoom, 1.)
            top_corners = tf.random.uniform(
                [batch_size, 2], 0, 1. - widths[:, None])
            bottom_corners = top_corners + widths[:, None]
            boxes = tf.concat((top_corners, bottom_corners), axis=1)

            x = tf.image.crop_and_resize(
                x, boxes,
                box_indices=tf.range(batch_size),
                crop_size=input_shape[0:2])

        if noise is not None:
            x = x + tf.random.normal(tf.shape(x), stddev=noise)

        return x, y
    
    return augment


if __name__ == '__main__':

    @scriptify
    def main(
        dataset,
        architecture,
        epochs=100,
        tuning_epochs=20,
        batch_size=128,
        heuristic='preserve',
        learning_rate=1e-3,
        learning_rate_schedule='fixed',
        tuning_lr=1e-5,
        tuning_lr_schdule='linear_dropoff',
        no_fine_tuning=False,
        gpu=0,
    ):
        # Select the GPU and allow memory growth to avoid taking all the RAM.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        device = gpus[gpu]

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        if 'MODEL_SAVE_DIR' in os.environ:
            model_path = os.environ['MODEL_SAVE_DIR']

            if not model_path.endswith('/'):
                model_path += '/'
        else:
            raise RuntimeError(
                'please set the "MODEL_SAVE_DIR" environment variable')

        # Get the data.
        if dataset.startswith('synth'):
            if 'SYNTHETIC_DATA_DIR' in os.environ:
                path = os.environ['SYNTHETIC_DATA_DIR']

                if not path.endswith('/'):
                    path += '/'
            else:
                raise RuntimeError(
                    'please set the "SYNTHETIC_DATA_DIR" environment variable')

            data = SyntheticOrderingPropertyData.load(path + dataset)

            properties = data.properties
            input_shape = data.input_shape
            num_classes = data.num_classes
            num_test_points = len(data.x_te)

            # Convert to tensorflow dataset.
            train = (Dataset.from_tensor_slices((data.x_tr, data.y_tr))
                .batch(batch_size))
            test = (Dataset.from_tensor_slices((data.x_te, data.y_te))
                .batch(batch_size))

        elif dataset == 'cifar100':
            tfds_dir = (
                os.environ['TFDS_DIR'] if 'TFDS_DIR' in os.environ else None)

            (train, test), metadata = tfds.load(
                'cifar100',
                data_dir=tfds_dir,
                split=['train', 'test'], 
                with_info=True, 
                shuffle_files=True, 
                as_supervised=True)

            train = (train
                .map(
                    lambda x,y: (tf.cast(x, 'float32') / 255., y), 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=False)
                .cache()
                .batch(batch_size)
                .map(
                    data_augmentation(
                        saturation=None, contrast=None, zoom=0.25), 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=False)
                .prefetch(tf.data.experimental.AUTOTUNE))

            test = (test
                .map(
                    lambda x,y: (tf.cast(x, 'float32') / 255., y), 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=False)
                .cache()
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE))

            properties = get_superclass_properties()
            input_shape = metadata.features['image'].shape
            num_classes = metadata.features['label'].num_classes
            num_test_points = 10000

        else:
            raise ValueError(f'unrecognized dataset: {dataset}')

        # Get the architecture.
        try:
            _orig_architecture = architecture
            params = '{}'

            if '.' in architecture:
                architecture, params = architecture.split('.', 1)

            x, y = getattr(architectures, architecture)(
                input_shape, num_classes, **json.loads(params))

        except:
            raise ValueError(f'unknown architecture: {_orig_architecture}')

        # Construct the model.
        f = Model(x, y)

        y_fixed = GeneralOrderingRepair(properties, heuristic)([x, y])

        f_self_repair = Model(x, y_fixed)

        y_checked = GeneralOrderingRepair(
            properties, heuristic, check_only=True)([x, y])

        f_checker = Model(x, y_checked)

        # Train the model.
        #   - Begin with the standard model.
        #   - Next, measure what happens when we apply the fix to the pre-
        #     trained model without additional training.
        #   - Finally, measure what happens after fine tuning an additional
        #     number of epochs with the repair layer.

        f.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(lr=learning_rate),
            metrics='acc')
        f_self_repair.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=SGD(lr=tuning_lr),
            metrics=['acc', fraction_in_violation])
        f_checker.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(lr=learning_rate),
            metrics=['acc', fraction_in_violation])

        # Train standard model.
        print('\ntraining standard model...')
        start_time = time()

        f.fit(
            train,
            epochs=epochs,
            callbacks=[LrScheduler(learning_rate_schedule)])

        train_time_per_epoch_std = (time() - start_time) / epochs

        # Evaluate standard model.
        start_time = time()

        _, test_acc_unsafe_std = f.evaluate(test)

        test_time_per_instance_std = (time() - start_time) / num_test_points

        # Evaluate standard model considering safety.
        _, test_safe_acc_std, test_violations_std = f_checker.evaluate(test)

        # Evaluate self-repairing model before fine-tuning.
        print('\nadding self-repair...')

        # Dry run first to "warm up," since this model hasn't been run yet.
        f_self_repair.evaluate(test, verbose=False)

        start_time = time()

        _, test_safe_acc_sr, test_violations_sr = f_self_repair.evaluate(test)

        test_time_per_instance_sr = (time() - start_time) / num_test_points

        if no_fine_tuning:
            return {
                'test_violations_std': float(test_violations_std),
                'test_acc_unsafe_std': float(test_acc_unsafe_std),
                'test_safe_acc_std': float(test_safe_acc_std),
                'test_safe_acc_sr': float(test_safe_acc_sr),
                'train_time_per_epoch_std': float(train_time_per_epoch_std),
                'test_time_per_instance_std': float(test_time_per_instance_std),
                'test_time_per_instance_sr': float(test_time_per_instance_sr),
                'test_violations_sr': float(test_violations_sr),
                'model_location': model_location,
            } 

        # Fine-tune the self-repairing model.
        print('\nfine tuning self-repairing model...')
        start_time = time()

        f_self_repair.fit(
            train,
            epochs=tuning_epochs,
            callbacks=[LrScheduler(tuning_lr_schdule)])

        train_time_per_epoch_sr = (time() - start_time) / tuning_epochs

        # Evaluate self-repairing model after fine-tuning.
        _, test_safe_acc_ft, test_violations_ft = f_self_repair.evaluate(test)

        # Save the model.
        random_name = blake2b(
            np.random.rand(1).tobytes(), digest_size=4).hexdigest()
        model_location = f'{model_path}{random_name}.h5'

        f.save(model_location)

        return {
            'test_violations_std': float(test_violations_std),
            'test_acc_unsafe_std': float(test_acc_unsafe_std),
            'test_safe_acc_std': float(test_safe_acc_std),
            'test_safe_acc_sr': float(test_safe_acc_sr),
            'test_safe_acc_ft': float(test_safe_acc_ft),
            'train_time_per_epoch_std': float(train_time_per_epoch_std),
            'train_time_per_epoch_sr': float(train_time_per_epoch_sr),
            'test_time_per_instance_std': float(test_time_per_instance_std),
            'test_time_per_instance_sr': float(test_time_per_instance_sr),
            'test_violations_sr': float(test_violations_sr),
            'test_violations_ft': float(test_violations_ft),
            'model_location': model_location,
        }
