#!/usr/bin/env python
# coding: utf-8


from audioop import avg
import io
from operator import ge
import os, sys
from re import M
from xml.dom import HIERARCHY_REQUEST_ERR
from cv2 import log
sys.path.append(os.path.abspath("."))
import tensorflow as tf
import time
from datetime import datetime as _dt
import matplotlib.pyplot as plt
from math import floor
import numpy as np

from .MODEL_CLASS import MODEL
from .Model_Critic import CRITIC
from .Model_Translator import GENERATOR

from util import img_functions


class PokeWGAN:
    def __init__(self, image_shape, output_dir, log_tiling = (4, 4), max_critic_filters=2048, plot_scaling=1):
        self.image_shape = image_shape
        if image_shape[-1] < 3:
            raise(ValueError("Image must have minimum dimensions of 3"))
        elif(image_shape[-1] == 3):
            self.get_hue = self._get_hue
        elif(image_shape[-1] == 4):
            self.get_hue = self._get_hue_alpha
        elif image_shape[-1] > 4:

            raise(NotImplementedError("Max Dim > 4 Images NYI"))

        self.birthday =_dt.now().strftime("%m_%d/%H")
        self.output_dir = os.path.join(output_dir, self.birthday)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("Building Generator!")

        self.generator = self._build_generator(self.image_shape)
        self.generator.summary()
        print("Building Critic!")
        self.critic = self._build_critic(self.image_shape, max_critic_filters)
        self.critic.summary()

        self.LAMBDA_GP = 10
        self.LAMBDA_REC = 10
        self.LAMBDA_HUE = 50
        self.train_g_every = 5

        self.gen_loss_keys = ['adv', 'rec', 'hue']
        self.critic_loss_keys = ['adv', 'gp']

        self.plot_scaling = plot_scaling


        self.checkpoint_folder = os.path.join(self.output_dir, 'checkpoints')

        self.checkpoint = tf.train.Checkpoint(
            generator = self.generator.model,
            critic = self.critic.model,
            g_opt = self.generator.optimizer,
            d_opt = self.critic.optimizer
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, os.path.join(self.checkpoint_folder, 'checkpoint'), max_to_keep=5)
        self.n_preds, self.m_preds= log_tiling
        self.seed = tf.random.uniform([self.n_preds*self.m_preds], 0.0, 1.0)
        self.epoch, self.epochs = 0, 1

    def _build_generator(self, image_shape):
        return GENERATOR(image_shape[:-1], image_shape[-1], image_shape[-1], self.output_dir, 1, lr=self._get_lr)
    def _build_critic(self, input_shape, max_filters):
        return CRITIC(input_shape, self.output_dir, max_filters=max_filters, lr=self._get_lr)

    def _get_lr(self):
        computed = 0.0001-((self.epoch+1)-(self.epochs/2))*(0.0001/(self.epochs/2))
        return tf.clip_by_value(computed, 0.0, 0.0001)

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

    def get_target_label(self, cls):
        target = tf.random.shuffle(cls)[0]
        target = tf.repeat([target], [tf.shape(cls)[0]], axis=0)
        return target

    def _get_hue(self, images):
        images = images*0.5+0.5
        avg_hues = img_functions.get_avg_hue(images)
        return avg_hues
    
    def _get_hue_alpha(self, images):
        images = images*0.5+0.5
        avg_hues = img_functions.get_avg_hue_alpha(images)
        return avg_hues

    def _train_step_critic(self, images, target_hues):
        with tf.GradientTape() as critic_tape:
            generated = self.generator([images, target_hues], training=True)

            real_output = self.critic(images, training=True)
            fake_output = self.critic(generated, training=True)

            gp = self.gradient_penalty(images, generated)*self.LAMBDA_GP
            adv_loss = tf.reduce_mean(fake_output - real_output)

            critic_loss = adv_loss + gp
        gradients_of_critic = critic_tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients_of_critic, self.critic.model.trainable_variables))

        return adv_loss, gp

    def _train_step_generator(self, images, avg_hues, target_hues):
        with tf.GradientTape() as generator_tape:
            generated = self.generator([images, target_hues], training=True)
            fake_output = self.critic(generated, training=True)

            hue_gen = self.get_hue(generated)
            hue_err = self.hue_loss(target_hues, hue_gen)*self.LAMBDA_HUE

            rec_img = self.generator([generated, avg_hues], training=True)

            rec_err = self.rec_loss(images, rec_img)*self.LAMBDA_REC

            adv_loss = tf.reduce_mean(-fake_output)

            gen_loss = adv_loss + rec_err + hue_err

        gradients_of_generator = generator_tape.gradient(gen_loss, self.generator.model.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.model.trainable_variables))

        return adv_loss, rec_err, hue_err

    def _train_step(self, images, batch):

        avg_hues  = self.get_hue(images)
        target_hues = self.get_target_label(avg_hues)
        critic_loss = self._train_step_critic(images, target_hues)
        gen_loss = self._train_step_generator(images, avg_hues, target_hues) if batch%self.train_g_every == 0 else [-1 for i in self.gen_loss_keys]

        return gen_loss, critic_loss

    def hue_loss (self, hue_real, hue_pred):
        return tf.reduce_mean(tf.math.square(img_functions.tf_angDist(hue_real, hue_pred, 1.0)))
    
    def rec_loss (self, real_img, fake_img):
        return tf.reduce_mean(tf.abs(tf.subtract(real_img, fake_img)))

    def train(self, img_ds, epochs, start_epoch = 0, generate_freq=1, checkpoint_freq=10):
        self.epochs = epochs
        ds_iter = iter(img_ds)
        log_set = next(ds_iter)
        log_hues = self.get_hue(log_set)
        seed_hues = tf.random.shuffle(log_hues)
        if log_set.shape[0] < self.n_preds*self.m_preds:
            raise(ValueError('Batch Size of Set to Log Too Small'))
        else:
            log_set = log_set[:self.n_preds*self.m_preds]

        total_gen_loss = tf.keras.metrics.Mean('total_gen_loss', dtype=tf.float32)
        gen_losses = []
        for i in self.gen_loss_keys:
            gen_losses.append(tf.keras.metrics.Mean('gen_'+i, dtype=tf.float32))

        #d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp
        total_critic_loss = tf.keras.metrics.Mean('total_critic_loss', dtype=tf.float32)
        critic_losses = []
        for i in self.critic_loss_keys:
            critic_losses.append(tf.keras.metrics.Mean('critic_'+i, dtype=tf.float32))

        for epoch in range(start_epoch, self.epochs):
            self.epoch = epoch
            start = time.time()
            localtime = time.localtime(start)
            print('Epoch {:04d} started at {}:{:02d}:{:02d}'
                  .format(epoch+1, localtime[3], localtime[4], localtime[5], 0),
                  end="\r", flush=True
                 )
            for i, image_batch in img_ds.enumerate():
                gen_loss_batch, critic_loss_batch = self._train_step(image_batch, i)
                print('Epoch {:04d} running at {}:{:02d}:{:02d} (Batch {} done!)'.format(
                    epoch+1, localtime[3], localtime[4], localtime[5], i),
                    end="\r", flush=True
                )
                if i%self.train_g_every == 0: 
                    for j in range(len(gen_loss_batch)):
                        gen_losses[j](gen_loss_batch[j])
                    total_gen_loss(tf.math.reduce_sum(gen_loss_batch))
                for j in range(len(critic_loss_batch)):
                    critic_losses[j](critic_loss_batch[j])
                total_critic_loss(tf.math.reduce_sum(critic_loss_batch))
                last_batch=i

            if ((epoch+1) % checkpoint_freq) == 0:
                self.checkpoint_manager.save()
            print('Time for Epoch {:04d} is {:.0f} seconds (Total {} batches completed)                '.format(epoch+1, time.time()-start, last_batch))
            print('\tGenerator Loss: {}\n\tDiscriminator Loss: {}'.format(
                    total_gen_loss.result(),
                    total_critic_loss.result()
            ))
            if ((epoch+1) % generate_freq) == 0:
                self.generate_and_log_imgs(log_set, epoch, seed=seed_hues)
                self._log_imgs(log_set, log_hues, epoch, f'Base')
            with self.generator.logger.as_default():
                tf.summary.scalar('loss', total_gen_loss.result(), step=epoch)
                for i in range(len(gen_losses)):
                    tf.summary.scalar(f'loss_{self.gen_loss_keys[i]}', gen_losses[i].result(), step=epoch)
            with self.critic.logger.as_default():
                tf.summary.scalar('loss', total_critic_loss.result(), step=epoch)
                for i in range(len(critic_losses)):
                    tf.summary.scalar(f'loss_{self.critic_loss_keys[i]}', critic_losses[i].result(), step=epoch)
            total_gen_loss.reset_states()
            total_critic_loss.reset_states()
            for i in range(len(critic_losses)):
                critic_losses[i].reset_states()
            for i in range(len(gen_losses)):
                gen_losses[i].reset_states()
        self.save_models()

    
    def generate_and_log_imgs(self, log_set, epoch, seed=None):
        if seed is None:
            seed = self.seed
        predictions = self.generator([log_set, seed], training=False)
        self._log_imgs(predictions, seed, epoch, f'Prediction', do_err=True)
    def _log_imgs(self, images, labels, epoch, log_str='', do_err = False, denorm=True):
        if denorm:
            images = images*0.5+0.5
        dpi = 100.
        w_pad = 2/72.
        h_pad = 2/72.
        plot_width = self.plot_scaling*(self.image_shape[-3]+2*w_pad*self.image_shape[-3])/dpi
        plot_height = self.plot_scaling*(self.image_shape[-2]+2*h_pad*self.image_shape[-2])/dpi
        fig, ax = plt.subplots(self.n_preds, self.m_preds, figsize=(plot_width*(self.m_preds+1), plot_height*(self.n_preds+1)), dpi=dpi)
        if do_err:
            hues = self.get_hue(images)
            error = img_functions.tf_angDist(labels, hues, 1.0)
        for i in range(self.n_preds):
            for j in range(self.m_preds):
                ax[i, j].imshow((images[i*self.m_preds+j, :, :, :].numpy()))
                if do_err:
                    ax[i, j].set_title(f'Error: {error[i*self.m_preds+j]:0.3f}')
                else:
                    ax[i, j].set_title(f'Hue: {labels[i*self.m_preds+j].numpy()*360:0.2f}')
                ax[i, j].axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        print('\t logging images for this epoch')
        with self.generator.logger.as_default():
            tf.summary.image(
                "Epoch {} Images".format(log_str),
                images, max_outputs=self.n_preds*self.m_preds, step=epoch
            )
            tf.summary.image(
                "_Epoch {} Image (concatenated)".format(log_str),
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
