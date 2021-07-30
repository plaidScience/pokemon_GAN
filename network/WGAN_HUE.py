import io
import os, sys
sys.path.append(os.path.abspath("."))
import tensorflow as tf

from datetime import datetime as _dt

from math import floor

from networks.BASE_MODEL_CLASS import _Base_Model_Class

class Generator(_Base_Model_Class):
    def __init__(self, input_shape, output_shape, output_dir):
        super(Generator, self).__init__(output_dir, name='gen', output_shape)
    def _build_model(self, input_shape=(100), output_shape=(28, 28, 1)):
        model = tf.keras.Sequential()
        #if the length of the input shape is greater than 1
        #     first downsample the model to be one
        if len(input_shape) > 1:
            model.add(tf.keras.layers.Conv2D(
                input_shape[-1]*64, (5,5),
                strides=(2,2), padding='same',
                input_shape = (
                    input_shape[-3],
                    input_shape[-2],
                    input_shape[-1]
                )
            ))
            model.add(tf.keras.layers.LeakyReLU())
            model.add(tf.keras.layers.Dropout(0.3))

            model.add(tf.keras.layers.Conv2D(input_shape[-1]*128, (5,5), strides=(2,2), padding='same'))
            model.add(tf.keras.layers.LeakyReLU())
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(output_shape[-1]*output_shape[-2]*output_shape[-3]*16, use_bias=False))
        else:
            model.add(tf.keras.layers.Dense(output_shape[-1]*output_shape[-2]*output_shape[-3]*16, use_bias=False, input_shape=(input_shape[-1]))

        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Reshape((floor(output_shape[-3]/4), floor(output_shape[-2]/4), output_shape[-1]*256)))

        model.add(tf.keras.layers.Conv2DTranspose(output_shape[-3]*128, (5,5), strides=(1,1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(output_shape[-3]*64, (5,5), strides=(2,2), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(output_shape[-3], (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, output_shape[-1], output_shape[-2], output_shape[-3])

        return model


    def _build_optimizer(self):
        return tf.keras.optimizers.RMSprop(lr=0.00005)

    def _build_loss(self):
        return tf.keras.losses.Mean()

class Critic(_Base_Model_Class):
    def __init__(self, input_shape, output_dir):
        super(Critic, self).__init__(output_dir, name='critic', input_shape)
    def _build_model(self, input_shape=(28, 28, 1)):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(input_shape[-3]*64, (5, 5), strides=(2, 2), padding='same', input_shape=(input_shape[-3], input_shape[-2], input_shape[-1])))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))


        model.add(tf.keras.layers.Conv2D(input_shape[-3]*128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))

        return model


    def _build_optimizer(self,):
        return tf.keras.optimizers.RMSprop(lr=0.00005)

    def _build_loss(self):
        return tf.keras.losses.Mean()


class HueWGAN:
    def __init__(self, input_shape, output_dir):
        self.birthday =_dt.now().strftime("%m_%d/%H")
        self.output_dir = os.path.join(output_dir, self.birthday)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.generator = self._build_generator(input_shape)
        self.generator.summary()
        self.critic = self._build_critic(input_shape)
        self.critic.summary()

        self.checkpoint_folder = os.path.join(self.output_dir, 'checkpoints')

        self.checkpoint = tf.train.Checkpoint(
            generator = self.generator.model,
            critic = self.critic.model,
            g_opt = self.generator.optimizer,
            d_opt = self.critic.optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, os.path.join(self.checkpoint_folder, 'checkpoint') max_to_keep=5)

    def _build_generator(self, shape):
        return Generator(shape, self.output_dir)
    def _build_critic(self, shape):
        return Critic(shape, self.output_dir)
