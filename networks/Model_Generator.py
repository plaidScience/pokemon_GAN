import tensorflow as tf
from .MODEL_CLASS import MODEL
from .NN_Layers import InstanceNormalization, ResnetBlock


class GENERATOR(MODEL):
# generator adapted from starGAN
    def __init__(self, input_shape, output_shape, output_dir, lr=0.0001, name='generator'):
        super(GENERATOR, self).__init__(
            output_dir,
            name=name,
            model_args={'input_shape':input_shape, 'output_shape':output_shape, 'n_resnet':6},
            optimizer_args={'lr':lr, 'beta_1':0.5, 'beta_2':0.999}
        )

    def _build_model(self, input_shape=None, output_shape=[None, None, 3], n_resnet=6):

        #inputs
        inp = tf.keras.Input(shape=input_shape)

        dense = tf.keras.layers.Dense((output_shape[-3]//16)*(output_shape[-2]//16)*32, use_bias=False)(inp)
        d_act = tf.keras.layers.ReLU()(dense)

        reshaped = tf.keras.layers.Reshape((output_shape[-3]//16, output_shape[-2]//16, 32))(d_act)



        padded_1 = tf.keras.layers.ZeroPadding2D(3)(reshaped)
        conv_1 = tf.keras.layers.Conv2D(64, (7,7), strides=1, padding='valid')(padded_1)
        norm_1 = InstanceNormalization()(conv_1)
        act_1 = tf.keras.layers.ReLU()(norm_1)
        
        deconv_1 = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=2, padding='same')(act_1)
        norm_2 = InstanceNormalization()(deconv_1)
        act_2 = tf.keras.layers.ReLU()(norm_2)
    
        deconv_2 = tf.keras.layers.Conv2DTranspose(256, (4,4), strides=2, padding='same')(act_2)
        norm_3 = InstanceNormalization()(deconv_2)
        act_3 = tf.keras.layers.ReLU()(norm_3)

        #resnet blocks
        resnet_blk = ResnetBlock(256, norm_type='instancenorm')(act_3)
        for _ in range(1, n_resnet):
            resnet_blk = ResnetBlock(256, norm_type='instancenorm')(resnet_blk)

        deconv_3 = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=2, padding='same')(resnet_blk)
        norm_4 = InstanceNormalization()(deconv_3)
        act_4 = tf.keras.layers.ReLU()(norm_4)

        deconv_4 = tf.keras.layers.Conv2DTranspose(64, (4,4), strides=2, padding='same')(act_4)
        norm_5 = InstanceNormalization()(deconv_4)
        act_5 = tf.keras.layers.ReLU()(norm_5)

        padded_2 = tf.keras.layers.ZeroPadding2D(3)(act_5)
        conv_2 = tf.keras.layers.Conv2D(3, (7,7), strides=1, padding='valid')(padded_2)
        norm_6 = InstanceNormalization()(conv_2)
        outp = tf.keras.layers.Activation('tanh')(norm_6)

        return tf.keras.Model(inputs=inp, outputs=outp, name=self.name)

    def _build_optimizer(self, lr=2e-4, beta_1=0.5, beta_2=0.999, **kwargs):
        return tf.keras.optimizers.Adam(lr, beta_1=beta_1, beta_2=beta_2, **kwargs)