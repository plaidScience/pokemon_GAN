import tensorflow as tf
from .MODEL_CLASS import MODEL
from .NN_Layers import InstanceNormalization, ResnetBlock


class GENERATOR(MODEL):
# generator adapted from starGAN
    def __init__(self, img_shape, inp_channels, outp_channels, output_dir, inp_classes=0, lr=0.0001, name='generator'):
        super(GENERATOR, self).__init__(
            output_dir,
            name=name,
            model_args={'img_shape':img_shape, 'input_channels':inp_channels, 'output_channels':outp_channels, 'input_classes':inp_classes, 'n_resnet':6},
            optimizer_args={'lr':lr, 'beta_1':0.5, 'beta_2':0.999}
        )

    def _build_model(self, img_shape=[None, None], input_channels=3, output_channels=3, input_classes=0, n_resnet=6):

        input_shape = tf.concat([img_shape, [input_channels]], axis=0)

        #inputs
        inp_img = tf.keras.Input(shape=input_shape)

        if input_classes > 0:
            inp_cls = tf.keras.Input(shape=[input_classes])

            labels = tf.keras.layers.RepeatVector(img_shape[-2]*img_shape[-1])(inp_cls)
            labels = tf.keras.layers.Reshape((img_shape[-2], img_shape[-1], input_classes)) (labels)
            inp = tf.keras.layers.Concatenate()([inp_img, labels])
            inp_model = [inp_img, inp_cls]


        else:
            inp = inp_img
            inp_model = inp_img

        padded_1 = tf.keras.layers.ZeroPadding2D(3)(inp)
        conv_1 = tf.keras.layers.Conv2D(64, (7,7), strides=1, padding='valid')(padded_1)
        norm_1 = InstanceNormalization()(conv_1)
        act_1 = tf.keras.layers.ReLU()(norm_1)

        #conv block 2
        padded_2 = tf.keras.layers.ZeroPadding2D(1)(act_1)
        conv_2 = tf.keras.layers.Conv2D(128, (4,4), strides=2, padding='valid')(padded_2)
        norm_2 = InstanceNormalization()(conv_2)
        act_2 = tf.keras.layers.ReLU()(norm_2)

        #conv block 3
        padded_3 = tf.keras.layers.ZeroPadding2D(1)(act_2)
        conv_3 = tf.keras.layers.Conv2D(256, (4,4), strides=2, padding='valid')(padded_3)
        norm_3 = InstanceNormalization()(conv_3)
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

        padded_4 = tf.keras.layers.ZeroPadding2D(3)(act_5)
        conv_4 = tf.keras.layers.Conv2D(output_channels, (7,7), strides=1, padding='valid')(padded_4)
        norm_6 = InstanceNormalization()(conv_4)
        outp = tf.keras.layers.Activation('tanh')(norm_6)

        return tf.keras.Model(inputs=inp_model, outputs=outp, name=self.name)

    def _build_optimizer(self, lr=2e-4, beta_1=0.5, beta_2=0.999, **kwargs):
        return tf.keras.optimizers.Adam(lr, beta_1=beta_1, beta_2=beta_2, **kwargs)