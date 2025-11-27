import tensorflow as tf
from cvnn.layers import ComplexBatchNormalization
from tensorflow.keras.initializers import Initializer

class ForceFloatInitializer(Initializer):
    def __init__(self, initializer):
        self.initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        # Force dtype to be float even if complex is requested
        if dtype and dtype.is_complex:
            dtype = tf.float32  # or tf.float64 depending on precision
        return self.initializer(shape, dtype=dtype, **kwargs)

    def get_config(self):
        return {"initializer": tf.keras.initializers.serialize(self.initializer)}

    @classmethod
    def from_config(cls, config):
        initializer = tf.keras.initializers.deserialize(config["initializer"])
        return cls(initializer=initializer)

class FixedComplexBatchNormalization(ComplexBatchNormalization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Wrap initializers to ensure they return float tensors
        if hasattr(self, 'moving_mean_initializer'):
             self.moving_mean_initializer = ForceFloatInitializer(self.moving_mean_initializer)
        if hasattr(self, 'moving_variance_initializer'):
             self.moving_variance_initializer = ForceFloatInitializer(self.moving_variance_initializer)
        if hasattr(self, 'beta_initializer'):
             self.beta_initializer = ForceFloatInitializer(self.beta_initializer)
        if hasattr(self, 'gamma_initializer'):
             self.gamma_initializer = ForceFloatInitializer(self.gamma_initializer)

    def build(self, input_shape):
        # Re-apply wrapper in build just in case super().build() resets them or if they are instantiated there
        # Although ComplexBatchNormalization likely uses them in build()
        if hasattr(self, 'moving_mean_initializer'):
             self.moving_mean_initializer = ForceFloatInitializer(self.moving_mean_initializer)
        if hasattr(self, 'moving_variance_initializer'):
             self.moving_variance_initializer = ForceFloatInitializer(self.moving_variance_initializer)
        if hasattr(self, 'beta_initializer'):
             self.beta_initializer = ForceFloatInitializer(self.beta_initializer)
        if hasattr(self, 'gamma_initializer'):
             self.gamma_initializer = ForceFloatInitializer(self.gamma_initializer)
        
        super().build(input_shape)
