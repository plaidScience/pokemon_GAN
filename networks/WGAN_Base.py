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

from .MODEL_CLASS import MODEL
from .Model_Critic import CRITIC
from .Model_Generator import GENERATOR


class PokeWGAN:
    def __init__(self, noise_dim, image_shape, output_dir):
        self.noise_dim = noise_dim
        self.image_shape = image_shape

        self.birthday =_dt.now().strftime("%m_%d/%H")
        self.output_dir = os.path.join(output_dir, self.birthday)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("Building Generator!")

        self.generator = self._build_generator(self.noise_dim, self.image_shape)
        self.generator.summary()
        print("Building Critic!")
        self.critic = self._build_critic(self.image_shape)
        self.critic.summary()
        self.LAMBDA_GP=10

        self.checkpoint_folder = os.path.join(self.output_dir, 'checkpoints')

        self.checkpoint = tf.train.Checkpoint(
            generator = self.generator.model,
            critic = self.critic.model,
            g_opt = self.generator.optimizer,
            d_opt = self.critic.optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, os.path.join(self.checkpoint_folder, 'checkpoint'), max_to_keep=5)
        self.n_preds, self.m_preds = 4, 4
        self.seed = tf.random.normal([self.n_preds*self.m_preds, self.noise_dim])

    def _build_generator(self, input_shape, output_shape):
        return GENERATOR([input_shape], output_shape, self.output_dir)
    def _build_critic(self, input_shape):
        return CRITIC(input_shape, self.output_dir)

    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform((real.shape[0], 1, 1, 1), minval=0.0, maxval=1.0)
        x_hat = (alpha*real + (1-alpha)*fake)
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.critic(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        d_regularizer = tf.reduce_mean(tf.square(ddx-1.0))
        return d_regularizer

    def _train_step_critic(self, images):
        g_input = tf.random.normal([images.shape[0], self.noise_dim])
        with tf.GradientTape() as critic_tape:
            generated = self.generator(g_input, training=True)

            real_output = self.critic(images, training=True)
            fake_output = self.critic(generated, training=True)

            gp = self.gradient_penalty(images, generated)

            critic_loss = tf.reduce_mean(fake_output - real_output) + gp*self.LAMBDA_GP
        gradients_of_critic = critic_tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients_of_critic, self.critic.model.trainable_variables))

        return critic_loss

    def _train_step_generator(self, batch_size):
        g_input = tf.random.normal([batch_size, self.noise_dim])
        with tf.GradientTape() as generator_tape:
            generated = self.generator(g_input, training=True)

            fake_output = self.critic(generated, training=True)

            gen_loss = -tf.reduce_mean(fake_output)

        gradients_of_generator = generator_tape.gradient(gen_loss, self.generator.model.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.model.trainable_variables))

        return gen_loss

    def _train_step(self, images, batch, train_g_every=5):
        g_input = tf.random.normal([images.shape[0], self.noise_dim])
        
        critic_loss = self._train_step_critic(images)
        gen_loss = self._train_step_generator(images.shape[0]) if batch%train_g_every == 0 else -1

        return gen_loss, critic_loss

    def train(self, img_ds, epochs, start_epoch = 0, generate_freq=1, checkpoint_freq=10):
        log_set = next(iter(img_ds))
        critic_loss = tf.keras.metrics.Mean("critic_loss", dtype=tf.float32)
        gen_loss = tf.keras.metrics.Mean("generator_loss", dtype=tf.float32)
        for epoch in range(start_epoch, epochs):
            start = time.time()
            localtime = time.localtime(start)
            print('Epoch {:04d} started at {}:{:02d}:{:02d} ({:.02f}% done)'
                  .format(epoch+1, localtime[3], localtime[4], localtime[5], 0),
                  end="\r", flush=True
                 )
            for i, image_batch in img_ds.enumerate():
                gen_loss_batch, critic_loss_batch = self._train_step(image_batch, 5)
                print('Epoch {:04d} running at {}:{:02d}:{:02d} (Batch {} done!)'.format(
                    epoch+1, localtime[3], localtime[4], localtime[5], i),
                    end="\r", flush=True
                )
                if gen_loss_batch is not None: gen_loss(gen_loss_batch)
                critic_loss(critic_loss_batch)

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
                tf.summary.scalar('loss', gen_loss.result(), step=epoch)
            with self.critic.logger.as_default():
                tf.summary.scalar('loss', critic_loss.result(), step=epoch)
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
        self.critic.save_model()
        self.generator.save_model()
    def load_models(self, path):
        self.critic.load_model(path)
        self.generator.load_model(path)
def main():
    pokeGAN = PokeWGAN((100,), (128, 128, 3), './OUTPUT/pokeGAN/')

if __name__ == '__main__':
    main()
