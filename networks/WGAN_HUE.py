#!/usr/bin/env python
# coding: utf-8


import io
import os, sys
sys.path.append(os.path.abspath("."))
import tensorflow as tf
import time
from datetime import datetime as _dt
import matplotlib.pyplot as plt
from math import floor
import numpy as np

from networks.BASE_MODEL_CLASS import _Base_Model_Class


class Generator(_Base_Model_Class):
    def __init__(self, input_shape, output_shape, output_dir):
        super(Generator, self).__init__(output_dir, name='gen', input_shape = input_shape, output_shape=output_shape)
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
                    input_shape[-1],
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
            model.add(tf.keras.layers.Dense(output_shape[-1]*output_shape[-2]*output_shape[-3]*16, use_bias=False, input_shape=(input_shape[-1],)))
        model.summary()
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Reshape((floor(output_shape[-3]/4), floor(output_shape[-2]/4), output_shape[-1]*256)))

        model.add(tf.keras.layers.Conv2DTranspose(output_shape[-1]*128, (5,5), strides=(1,1), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(output_shape[-1]*64, (5,5), strides=(2,2), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        model.add(tf.keras.layers.Conv2DTranspose(output_shape[-1], (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, output_shape[-3], output_shape[-2], output_shape[-1])

        return model


    def _build_optimizer(self):
        return tf.keras.optimizers.RMSprop(0.00005)

    def _build_loss(self):
        return lambda y: tf.math.reduce_mean(y)

class Critic(_Base_Model_Class):
    def __init__(self, input_shape, clip_value, output_dir):
        super(Critic, self).__init__(output_dir, name='critic', input_shape=input_shape, clip_value=clip_value)
    def _build_model(self, input_shape=(28, 28, 1), clip_value=0.01):
        constraint= tf.keras.constraints.MinMaxNorm(min_value=-clip_value, max_value=clip_value)
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(
            input_shape[-1]*64, (5, 5), strides=(2, 2),
            padding='same', input_shape=(input_shape[-3], input_shape[-2], input_shape[-1]),
            kernel_constraint = constraint
        ))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))


        model.add(tf.keras.layers.Conv2D(input_shape[-1]*128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))

        return model


    def _build_optimizer(self,):
        return tf.keras.optimizers.RMSprop(0.00005)

    def _build_loss(self):
        return lambda y: tf.math.reduce_mean(y)


class PokeWGAN:
    def __init__(self, input_shape, output_shape, output_dir):
        self.input_shape, self.output_shape = input_shape, output_shape

        self.birthday =_dt.now().strftime("%m_%d/%H")
        self.output_dir = os.path.join(output_dir, self.birthday)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("Building Generator!")

        self.generator = self._build_generator(self.input_shape, self.output_shape)
        self.generator.summary()
        print("Building Critic!")
        self.critic = self._build_critic(self.output_shape)
        self.critic.summary()

        self.checkpoint_folder = os.path.join(self.output_dir, 'checkpoints')

        self.checkpoint = tf.train.Checkpoint(
            generator = self.generator.model,
            critic = self.critic.model,
            g_opt = self.generator.optimizer,
            d_opt = self.critic.optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, os.path.join(self.checkpoint_folder, 'checkpoint'), max_to_keep=5)
        self.n_preds, self.m_preds = 4, 4
        self.seed = tf.random.normal([self.n_preds*self.m_preds, self.input_shape[-1]])

    def _build_generator(self, input_shape, output_shape):
        return Generator(input_shape, output_shape, self.output_dir)
    def _build_critic(self, input_shape, clip_value=0.01):
        return Critic(input_shape, clip_value, self.output_dir)

    def _train_step_critic(self, images):
        if len(self.input_shape) == 1:
            g_input = tf.random.normal([images.shape[0], self.input_shape[-1]])
        else:
            raise ValueError("Inputs with dimension > 1 NYI!")
        with tf.GradientTape() as critic_tape:
            generated = self.generator(g_input, training=True)

            real_output = self.critic(images, training=True)
            fake_output = self.critic(generated, training=True)

            critic_loss = self.critic.loss(fake_output - real_output)
        gradients_of_critic = critic_tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients_of_critic, self.critic.model.trainable_variables))

        return critic_loss

    def _train_step_generator(self, batch_size):
        if len(self.input_shape) == 1:
            g_input = tf.random.normal([batch_size, self.input_shape[-1]])
        else:
            raise ValueError("Inputs with dimension > 1 NYI!")
        with tf.GradientTape() as generator_tape:
            generated = self.generator(g_input, training=True)

            fake_output = self.critic(generated, training=True)

            gen_loss = self.generator.loss(-1*fake_output)

        gradients_of_generator = generator_tape.gradient(gen_loss, self.generator.model.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.model.trainable_variables))

        return gen_loss

    def _train_step(self, images, batch, train_g_every=5):
        if len(self.input_shape) == 1:
            g_input = tf.random.normal([images.shape[0], self.input_shape[-1]])
        else:
            raise ValueError("Inputs with dimension > 1 NYI!")
        with tf.GradientTape() as generator_tape, tf.GradientTape() as critic_tape:
            generated = self.generator(g_input, training=True)

            real_output = self.critic(images, training=True)
            fake_output = self.critic(generated, training=True)

            gen_loss = self.generator.loss(-1*tf.slice(fake_output, [0, 0], [floor(fake_output.shape[0]/5.0), 1]))
            critic_loss = self.critic.loss(fake_output - real_output)
        gradients_of_generator = generator_tape.gradient(gen_loss, self.generator.model.trainable_variables)
        gradients_of_critic = critic_tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.model.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(gradients_of_critic, self.critic.model.trainable_variables))
        return gen_loss, critic_loss

    def train(self, img_ds, epochs, start_epoch = 0, generate_freq=1, checkpoint_freq=10):
        log_set = next(iter(img_ds))
        critic_loss = tf.keras.metrics.Mean("critic_loss", dtype=tf.float32)
        gen_loss = tf.keras.metrics.Mean("generator_loss", dtype=tf.float32)
        j = None
        for epoch in range(start_epoch, epochs):
            start = time.time()
            localtime = time.localtime(start)
            print('Epoch {:04d} started at {}:{:02d}:{:02d} ({:.02f}% done)'
                  .format(epoch+1, localtime[3], localtime[4], localtime[5], 0),
                  end="\r", flush=True
                 )
            i = 0
            for image_batch in img_ds:
                gen_loss_batch, critic_loss_batch = self._train_step(image_batch, i)
                i+=1
                if j is None and i%10 == 0:
                    print('Epoch {:04d} running at {}:{:02d}:{:02d} (Batch {} done!)'.format(
                        epoch+1, localtime[3], localtime[4], localtime[5], i),
                        end="\r", flush=True
                    )
                elif j is not None:
                    print('Epoch {:04d} running at {}:{:02d}:{:02d} ({:.02f}% done!)'.format(
                        epoch+1, localtime[3], localtime[4], localtime[5], 100*i/j),
                        end="\r", flush=True
                    )
                if gen_loss_batch is not None: gen_loss(gen_loss_batch)
                critic_loss(critic_loss_batch)
            j = i-1

            if ((epoch+1) % checkpoint_freq) == 0:
                self.checkpoint_manager.save()
            print('Time for Epoch {:04d} is {:.0f} seconds                '.format(epoch+1, time.time()-start))
            print('\tGenerator Loss: {}\n\tDiscriminator Loss: {}'.format(
                    gen_loss.result(),
                    critic_loss.result()
            ))
            if ((epoch+1) % generate_freq) == 0:
                self.generate_and_log_imgs(epoch)
                self._log_imgs(log_set, epoch, 'Base')
            with self.generator.logger.as_default():
                tf.summary.scalar('generator_loss', gen_loss.result(), step=epoch)
            with self.critic.logger.as_default():
                tf.summary.scalar('critic_loss', critic_loss.result(), step=epoch)
            gen_loss.reset_states()
            critic_loss.reset_states()
        self.save_models()
    def generate_and_log_imgs(self, epoch):
        predictions = self.generator(self.seed, training=False)
        self._log_imgs(predictions, epoch, 'Prediction')
    def _log_imgs(self, images, epoch, log_str=''):
        fig = plt.figure(figsize=(self.n_preds, self.m_preds))
        for i in range(self.n_preds*self.m_preds):
            plt.subplot(self.n_preds, self.m_preds, i+1)
            plt.imshow((images[i, :, :, :]*127.5 + 127.5).numpy().astype(np.uint8))
            plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        print('\t logging images for this epoch')
        with self.generator.logger.as_default():
            tf.summary.image(
                "Epoch{} Images".format(log_str),
                images, max_outputs=self.n_preds*self.m_preds, step=epoch
            )
            tf.summary.image(
                "Epoch {} Image (concatenated)".format(log_str),
                image, max_outputs=self.n_preds*self.m_preds, step=epoch
            )
    def save_models(self):
        self.critic.save()
        self.generator.save()
def main():
    pokeGAN = PokeWGAN((100,), (68, 56, 3), './OUTPUT/pokeGAN/')

if __name__ == '__main__':
    main()
