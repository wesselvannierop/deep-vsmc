"""
Defining the State and StateSeries classes, which are used to store the state
of the filter at a given time step.

Important sources:
- https://www.tensorflow.org/guide/extension_type
- https://www.tensorflow.org/api_docs/python/tf/TensorArray
- https://www.tensorflow.org/guide/function
"""

import pickle
from collections import namedtuple

import jax
import keras
import tensorflow as tf
from jax.tree_util import register_pytree_node
from keras import ops
from usbmd.tensor_ops import extend_n_dims


class StateMethods:
    ATTRIBUTES = [
        "particles",
        "entropy",
        "log_weights",
        "weights",
        "log_likelihoods",
        "action",
        "ancestor_indices",
        "resampling_correction",
        "t",
        "ess",
    ]

    @property
    def batch_size(self):
        return ops.shape(self.particles)[self.batch_axis]

    @property
    def n_particles(self):
        return ops.shape(self.particles)[self.particle_axis]

    @property
    def n_particles_float(self):
        return ops.cast(self.n_particles, "float32")

    @property
    def dimension(self):
        return ops.shape(self.particles)[self.particle_axis + 1 :]

    @property
    def flat_particles(self):
        """Flatten the latent dimensions of the particles."""
        _shape = ops.shape(self.particles)[: -len(self.dimension)]
        return ops.reshape(self.particles, (*_shape, -1))

    def get_particle(self, mode: str):
        """Posterior estimate of the state."""
        if mode == "wm" or mode == "weighted_mean":
            return self.get_weighted_mean()
        elif mode == "ml" or mode == "most_likely":
            return self.get_ml_particles()
        elif mode == "mean" or mode == "average":
            return self.get_mean_particles()
        elif mode == "var" or mode == "variance":
            return self.get_var_particles()
        elif mode == "sample":
            return self.sample_particle()
        else:
            raise ValueError(f"Unknown mode {mode}")

    def get_weighted_mean(self):
        weights = extend_n_dims(self.weights, axis=-1, n_dims=len(self.dimension))
        weighted_mean = ops.sum(self.particles * weights, axis=self.particle_axis)
        return weighted_mean

    def get_ml_indices(self):
        return ops.argmax(self.weights, axis=self.particle_axis)

    def get_ml_particles(self):
        ml_particle_indices = self.get_ml_indices()
        ml_particle_indices = extend_n_dims(
            ml_particle_indices, axis=-1, n_dims=1 + len(self.dimension)
        )
        ml_particles = ops.take_along_axis(
            self.particles, ml_particle_indices, axis=self.particle_axis
        )
        ml_particles = ops.squeeze(ml_particles, axis=self.particle_axis)
        return ml_particles

    def get_mean_particles(self):
        """Assumes equal weights."""
        return ops.mean(self.particles, axis=self.particle_axis)

    def get_var_particles(self):
        """Calculate the weighted variance of the particles."""
        weights = extend_n_dims(self.weights, axis=-1, n_dims=len(self.dimension))
        weighted_mean = ops.sum(self.particles * weights, axis=self.particle_axis)
        weighted_mean = ops.expand_dims(weighted_mean, axis=self.particle_axis)

        variance = ops.sum(
            weights * (self.particles - weighted_mean) ** 2, axis=self.particle_axis
        )
        return variance

    def sample_particle(self):
        random_idx = self._sample_particle_idx()
        random_idx = extend_n_dims(random_idx, axis=-1, n_dims=len(self.dimension) + 1)
        particle = ops.take_along_axis(
            self.particles, random_idx, axis=self.particle_axis
        )
        return ops.squeeze(particle, axis=self.particle_axis)

    def call_method_on_all_attributes(self, method, *args, **kwargs):
        updated_attributes = {}
        for attr_name in self.ATTRIBUTES:
            attr = getattr(self, attr_name)
            updated_attributes[attr_name] = method(attr, *args, **kwargs)
        return updated_attributes

    @classmethod
    def load(cls, filepath, *args, **kwargs):
        with open(filepath, "rb") as file:
            obj = pickle.load(file, *args, **kwargs)

        return cls(**obj)

    def dump(self, filepath, *args, **kwargs):
        # Convert all attributes to numpy
        obj = self.call_method_on_all_attributes(ops.convert_to_numpy)

        # Dump dict with numpy arrays to file
        with open(str(filepath), "wb") as file:
            return pickle.dump(obj, file, *args, **kwargs)

    def evolve(self, **kwargs):
        # Because ExtensionType is immutable, we need to create a new object
        for attr_name in self.ATTRIBUTES:
            if attr_name not in kwargs:
                kwargs[attr_name] = getattr(self, attr_name)
        return self.__class__(**kwargs)

    def _flatten_func_state(self):
        # children must contain arrays & pytrees
        children = (
            self.particles,
            self.entropy,
            self.log_weights,
            self.weights,
            self.log_likelihoods,
            self.action,
            self.ancestor_indices,
            self.resampling_correction,
            self.t,
            self.ess,
        )

        aux_data = None  # aux_data must contain static, hashable data.
        return (children, aux_data)

    @classmethod
    def _unflatten_func_state(cls, aux_data, children):
        # Here we avoid `__init__` because it has extra logic we don't require:
        obj = object.__new__(cls)
        obj.particles = children[0]
        obj.entropy = children[1]
        obj.log_weights = children[2]
        obj.weights = children[3]
        obj.log_likelihoods = children[4]
        obj.action = children[5]
        obj.ancestor_indices = children[6]
        obj.resampling_correction = children[7]
        obj.t = children[8]
        obj.ess = children[9]
        return obj

    def _sample_particle_idx(self):
        # is implemented for State and StateSeries
        raise NotImplementedError


class State(StateMethods):
    def __init__(
        self,
        particles,
        entropy=None,
        log_weights=None,
        weights=None,
        log_likelihoods=None,
        ancestor_indices=None,
        resampling_correction=None,
        t=None,
        action=None,
        ess=None,
        **kwargs,
    ):
        self.particles = particles
        self.entropy = entropy
        self.log_weights = log_weights
        self.weights = weights
        self.log_likelihoods = log_likelihoods
        self.ancestor_indices = ancestor_indices
        self.resampling_correction = resampling_correction
        self.t = t
        self.action = action
        self.ess = ess

        # If flat_particles is provided, reshape and overwrite particles
        for kwarg in kwargs:
            if kwarg == "flat_particles":
                flat_particles = kwargs[kwarg]
                self.particles = ops.reshape(
                    flat_particles, (*ops.shape(flat_particles)[:-1], *self.dimension)
                )
            else:
                raise ValueError(f"Unknown keyword argument {kwarg}")

        # Set default values
        if self.entropy is None:
            self.entropy = ops.zeros(self.batch_size)
        if self.weights is None:
            self.weights = (
                ops.ones((self.batch_size, self.n_particles)) / self.n_particles_float
            )
        if self.log_weights is None:
            self.log_weights = ops.log(self.weights)
        if self.log_likelihoods is None:
            self.log_likelihoods = ops.zeros(self.batch_size)
        if self.ancestor_indices is None:
            ancestor_indices = ops.arange(self.n_particles)
            ancestor_indices = ops.tile(
                ops.expand_dims(ancestor_indices, 0), [self.batch_size, 1]
            )
            self.ancestor_indices = ancestor_indices
        if self.resampling_correction is None:
            self.resampling_correction = ops.zeros(self.batch_size)
        if self.t is None:
            self.t = ops.array(0, dtype="int32")
        if self.action is None:
            # NOTE: just set to some tensor, action should be set if it is used!
            self.action = ops.zeros(self.batch_size)
        if self.ess is None:
            self.ess = ops.zeros(self.batch_size) + self.n_particles_float

    @property
    def batch_axis(self):
        return 0

    @property
    def particle_axis(self):
        return 1

    def _sample_particle_idx(self):
        return keras.random.randint(
            (self.batch_size,), minval=0, maxval=self.n_particles
        )


class StateTF(State, tf.experimental.ExtensionType):
    particles: tf.Tensor
    entropy: tf.Tensor
    log_weights: tf.Tensor
    weights: tf.Tensor
    log_likelihoods: tf.Tensor
    ancestor_indices: tf.Tensor
    resampling_correction: tf.Tensor
    t: tf.Tensor
    action: tf.Tensor
    ess: tf.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def index_default_to_none(value, index: int):
    """Return the value at index if it exists, otherwise return None."""
    not_empty = value is not None and len(value) > index
    return value[index] if not_empty else None


class StateSeries(StateMethods):
    def __init__(self, state_array):
        self.particles = state_array.particles
        self.entropy = state_array.entropy
        self.log_weights = state_array.log_weights
        self.weights = state_array.weights
        self.log_weights = state_array.log_weights
        self.log_likelihoods = state_array.log_likelihoods
        self.ancestor_indices = state_array.ancestor_indices
        self.resampling_correction = state_array.resampling_correction
        self.t = state_array.t
        self.action = state_array.action
        self.ess = state_array.ess

    @property
    def time_axis(self):
        return 0

    @property
    def batch_axis(self):
        return 1

    @property
    def particle_axis(self):
        return 2

    @property
    def n_observations(self):
        return ops.shape(self.particles)[self.time_axis]

    def read(self, time: int):
        """Read the state at the given time step."""
        return State(
            particles=index_default_to_none(self.particles, time),
            entropy=index_default_to_none(self.entropy, time),
            log_weights=index_default_to_none(self.log_weights, time),
            weights=index_default_to_none(self.weights, time),
            log_likelihoods=index_default_to_none(self.log_likelihoods, time),
            ancestor_indices=index_default_to_none(self.ancestor_indices, time),
            resampling_correction=index_default_to_none(
                self.resampling_correction, time
            ),
            t=index_default_to_none(self.t, time),
            action=index_default_to_none(self.action, time),
            ess=index_default_to_none(self.ess, time),
        )

    def __getitem__(self, item: int):
        return self.read(item)

    def __repr__(self):
        return f"<StateSeries n_observations={self.n_observations}>"

    def _sample_particle_idx(self):
        return keras.random.randint(
            (self.n_observations, self.batch_size),
            minval=0,
            maxval=self.n_particles,
        )


class StateSeriesTF(StateSeries, tf.experimental.ExtensionType):
    particles: tf.Tensor
    entropy: tf.Tensor
    log_weights: tf.Tensor
    weights: tf.Tensor
    log_likelihoods: tf.Tensor
    ancestor_indices: tf.Tensor
    resampling_correction: tf.Tensor
    t: tf.Tensor
    action: tf.Tensor
    ess: tf.Tensor

    def __init__(self, state_array):
        self.particles = state_array.particles.stack()
        self.entropy = state_array.entropy.stack()
        self.log_weights = state_array.log_weights.stack()
        self.weights = state_array.weights.stack()
        self.log_weights = state_array.log_weights.stack()
        self.log_likelihoods = state_array.log_likelihoods.stack()
        self.ancestor_indices = state_array.ancestor_indices.stack()
        self.resampling_correction = state_array.resampling_correction.stack()
        self.t = state_array.t.stack()
        self.action = state_array.action.stack()
        self.ess = state_array.ess.stack()


StateArray = namedtuple("StateArray", StateMethods.ATTRIBUTES)


def write_state_array(state: State, n_observations=None, state_array=None, time=0):
    tas = []
    for attr_name in state.ATTRIBUTES:
        value = getattr(state, attr_name)

        if state_array is None:
            ta = create_tensor_array(value.shape, value.dtype)
        else:
            ta = getattr(state_array, attr_name)

        ta = ta.write(time, value)
        tas.append(ta)
    return StateArray(*tas)


def create_tensor_array(shape, dtype=tf.float32):
    return tf.TensorArray(
        dtype,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        element_shape=tf.TensorShape(shape),
    )


def write_state_array_prealloc(
    state: State, n_observations=None, state_array=None, time=0
):
    """Preallocate the state array and write the state at the given time step.
    Probably can replace the jax specific implementation with this one.
    """
    tas = []
    for attr_name in StateMethods.ATTRIBUTES:
        value = getattr(state, attr_name)

        if state_array is None:
            ta = create_tensor_array_prealloc(
                ops.shape(value), ops.dtype(value), n_observations
            )
        else:
            ta = getattr(state_array, attr_name)

        ta = ops.scatter_update(ta, indices=[[time]], updates=[value])
        tas.append(ta)
    return StateArray(*tas)


def create_tensor_array_prealloc(shape, dtype, n_observations):
    return ops.zeros((n_observations, *shape), dtype=dtype)


def write_state_array_jax(state: State, n_observations=None, state_array=None, time=0):
    tas = []
    for attr_name in state.ATTRIBUTES:
        value = getattr(state, attr_name)

        if state_array is None:
            ta = create_tensor_array_jax(value.shape, value.dtype, n_observations)
        else:
            ta = getattr(state_array, attr_name)

        ta = ta.at[time].set(value)
        tas.append(ta)
    return StateArray(*tas)


def create_tensor_array_jax(shape, dtype, n_observations):
    return jax.numpy.zeros((n_observations, *shape), dtype=dtype)


class KFState(State):
    # TODO: finalize the KFState class and support in the rest of the code
    def __init__(self, particles_cov, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.particles_cov = particles_cov


for cls in StateMethods.__subclasses__():
    register_pytree_node(cls, cls._flatten_func_state, cls._unflatten_func_state)
