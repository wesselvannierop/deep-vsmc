"""
Will load the correct tensorflow probability module based on the KERAS_BACKEND environment variable.
This is useful for compatibility with both TensorFlow and JAX backends.
"""

import os
import warnings

if os.environ["KERAS_BACKEND"] == "tensorflow":
    from tensorflow_probability import *
elif os.environ["KERAS_BACKEND"] == "jax":
    from tensorflow_probability.substrates.jax import *
else:
    warnings.warn(
        f"Unsupported KERAS_BACKEND for tfp: {os.environ['KERAS_BACKEND']} "
        "falling back to tensorflow"
    )
    from tensorflow_probability import *
