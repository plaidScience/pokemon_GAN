import tensorflow as tf
import os

class _Base_Model_Class():
    def __init__(self, output_dir, name='model', **model_args):
        self.output_dir = output_dir
        self.name = name
        self.model = self._build_model(**model_args)
        self.optimizer = self._build_optimizer()
        self.loss = self._build_loss()
        self.logger = tf.summary.create_file_writer(os.path.join(self.output_dir, 'logs/', '{}/'.format(self.name)))
    def __call__(input, **kwargs):
        return model(input, **kwargs)
    def save_model():
        self.model.save(os.path.join(self.output_dir, 'saved_models', '{}/'.format(self.name)))
    def load_model(save_folder, name=None):
        if name is None:
            name = self.name
        if os.path.exists(os.path.join(save_folder, name)):
            self.model = tf.keras.models.load_model(os.path.join(save_folder, name))
        else:
            raise OSError("Model Save Path Not found in Directory")
    def summary(self):
        self.model.summary()
    def _build_model(**args):
        #to be written over by subclass
        return tf.keras.Sequential()
    def _build_optimizer():
        #to be written over by subclass
        return tf.keras.optimizers.Optimizer('model_optimizer')
    def _build_loss():
        #to be written over by subclass
        return tf.keras.losses.Loss()
