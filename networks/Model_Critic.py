import tensorflow as tf
from .MODEL_CLASS import MODEL
from .NN_Layers import InstanceNormalization, DownsampleBlock


class CRITIC(MODEL):
# critic adapted from starGAN
    def __init__(self, input_shape, output_dir, max_filters=2048, up_to_max=True, lr=0.0001, name='critic'):
        super(CRITIC, self).__init__(
            output_dir,
            name=name,
            model_args={'input_shape':input_shape, 'max_filters':max_filters, 'up_to_max':up_to_max},
            optimizer_args={'lr':lr, 'beta_1':0.5, 'beta_2':0.999}
        )

    def _build_model(self, input_shape=[None, None, 3], max_filters=2048, up_to_max=True):

        norm_type='none'

        inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
        x = inp


        w = input_shape[-3]
        h = input_shape[-2]
        filters = 64

        while (w > 1 and h > 1 and filters <= max_filters):
            x = tf.keras.layers.ZeroPadding2D(1)(x)
            x = tf.keras.layers.Conv2D(filters, (4,4), strides=2, padding='valid')(x)
            x= tf.keras.layers.LeakyReLU()(x)

            w = w//2
            h = h//2
            filters = filters*2
        
        while (filters <= max_filters and up_to_max==True):
            x = tf.keras.layers.Conv2D(filters, (4,4), strides=1, padding='same')(x)
            x= tf.keras.layers.LeakyReLU()(x)
            filters = filters*2

        x1=tf.keras.layers.ZeroPadding2D(1)(x)
        src = tf.keras.layers.Conv2D(1, (3, 3), strides=1, padding='valid')(x1)

        return tf.keras.Model(inputs=inp, outputs=src, name=self.name)

    def _build_optimizer(self, lr=2e-4, beta_1=0.5, beta_2=0.999, **kwargs):
        return tf.keras.optimizers.Adam(lr, beta_1=beta_1, beta_2=beta_2, **kwargs)
