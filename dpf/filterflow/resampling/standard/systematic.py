import keras
from keras import ops

from dpf.filterflow.resampling.standard.base import StandardResamplerBase
from dpf.prob import stateless_to_keras_seed


def _systematic_spacings(n_particles, batch_size, seed):
    """Generate non decreasing numbers x_i between [0, 1].

    Args:
        n_particles (int): Number of particles.
        batch_size (int): Batch size.
        seed (Tuple[int]): Stateless random seed.

    Returns:
        Tensor: Spacings.
    """
    float_n_particles = ops.cast(n_particles, "float32")
    seed = stateless_to_keras_seed(seed)
    z = keras.random.uniform((batch_size, 1), seed=seed)
    z = z + ops.reshape(
        ops.linspace(0.0, float_n_particles - 1.0, n_particles), [1, -1]
    )
    return z / float_n_particles


class SystematicResampler(StandardResamplerBase):
    def __init__(self, on_log=True):
        super().__init__(on_log)

    @staticmethod
    def _get_spacings(n_particles, batch_size, seed):
        return _systematic_spacings(n_particles, batch_size, seed)
