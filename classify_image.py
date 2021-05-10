import time
import statistics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.eager import context

import numpy as np

class ClassifyImage:

    def __init__(self):
        self.epochs = 1
        self.num_measures = 5
        self.num_warmup_skips = 1

    def classifyImageCPU(self, parameters):
        self.reset()
        self.set_parameters(parameters)
        self.build_model()

        times = []
        for i in range (0, self.num_measures):
            #tf.profiler.experimental.start('logdir')
            t1_start = time.perf_counter()
            self.run()
            t1_stop = time.perf_counter()
            #tf.profiler.experimental.stop()
            if i >= self.num_warmup_skips: # first few rounds are warmup only
                times.append(round((t1_stop-t1_start) * 1000))

        t_mean = round(statistics.mean(times))
        t_stdev = round(statistics.stdev(times))
        print("DEBUG: " + str(times) + " -> " + str(t_mean) + "±" + str(t_stdev) + " ms")

        return t_mean, t_stdev

    def reset(self):
        context._context = None
        context._create_context()
        tf.random.set_seed(130)
        np.random.seed(130)


    def set_parameters(self, parameters):
        optimizer_options = tf.config.optimizer.get_experimental_options()

        for key in ["layout_optimizer", "constant_folding", "shape_optimization", "remapping", "arithmetic_optimization",
                    "dependency_optimization", "loop_optimization", "function_optimization", "debug_stripper",
                    "disable_model_pruning", "scoped_allocator_optimization", "pin_to_host_optimization",
                    "implementation_selector", "auto_mixed_precision", "disable_meta_optimizer"]:
            param_key = "tf.config.optimizer.set_experimental_options." + key
            if param_key in parameters:
                optimizer_options[key] = parameters[param_key]

        tf.config.optimizer.set_experimental_options(optimizer_options)
        print("DEBUG: optimizer.experimental_options: " + str(tf.config.optimizer.get_experimental_options()))

        tf.config.threading.set_inter_op_parallelism_threads(
            parameters.get("tf.config.threading.inter_op_parallelism", tf.config.threading.get_inter_op_parallelism_threads()))
        tf.config.threading.set_intra_op_parallelism_threads(
            parameters.get("tf.config.threading.intra_op_parallelism", tf.config.threading.get_intra_op_parallelism_threads()))
        print("DEBUG: threading.inter_op_parallelism_threads: " + str(tf.config.threading.get_inter_op_parallelism_threads()))
        print("DEBUG: threading.intra_op_parallelism_threads: " + str(tf.config.threading.get_intra_op_parallelism_threads()))



    def build_model(self):
        image_size = (180, 180)
        batch_size = 32

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "PetImages",
            validation_split=0.2,
            subset="training",
            seed=130,
            image_size=image_size,
            batch_size=batch_size,
        )
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "PetImages",
            validation_split=0.2,
            subset="validation",
            seed=130,
            image_size=image_size,
            batch_size=batch_size,
        )

        #data_augmentation = keras.Sequential(
        #    [
        #        layers.experimental.preprocessing.RandomFlip("horizontal"),
        #        layers.experimental.preprocessing.RandomRotation(0.1),
        #    ]
        #)

        #augmented_train_ds = self.train_ds.map(
        #  lambda x, y: (data_augmentation(x, training=True), y))

        self.train_ds = self.train_ds.prefetch(buffer_size=32)
        self.val_ds = self.val_ds.prefetch(buffer_size=32)

        def make_model(input_shape, num_classes):
            inputs = keras.Input(shape=input_shape)
            # Image augmentation block
            #x = data_augmentation(inputs)
            x = inputs

            # Entry block
            x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
            x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.Conv2D(64, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            previous_block_activation = x  # Set aside residual

            for size in [128, 256, 512, 728]:
                x = layers.Activation("relu")(x)
                x = layers.SeparableConv2D(size, 3, padding="same")(x)
                x = layers.BatchNormalization()(x)

                x = layers.Activation("relu")(x)
                x = layers.SeparableConv2D(size, 3, padding="same")(x)
                x = layers.BatchNormalization()(x)

                x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

                # Project residual
                residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                    previous_block_activation
                )
                x = layers.add([x, residual])  # Add back residual
                previous_block_activation = x  # Set aside next residual

            x = layers.SeparableConv2D(1024, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            x = layers.GlobalAveragePooling2D()(x)
            if num_classes == 2:
                activation = "sigmoid"
                units = 1
            else:
                activation = "softmax"
                units = num_classes

            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(units, activation=activation)(x)
            return keras.Model(inputs, outputs)


        self.model = make_model(input_shape=image_size + (3,), num_classes=2)
        keras.utils.plot_model(self.model, show_shapes=True)

        self.callbacks = [
            keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
        ]
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def run(self):
        history = self.model.fit(self.train_ds, epochs=self.epochs, callbacks=self.callbacks, validation_data=self.val_ds)
        #history.history['accuracy'], history.history['val_accuracy']
