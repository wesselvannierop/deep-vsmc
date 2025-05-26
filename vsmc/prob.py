import keras

import vsmc.tfp_wrapper as tfp
from vsmc.keras_helpers import deserialize

tfd = tfp.distributions


class StatelessSeedGenerator(keras.random.SeedGenerator):
    def __init__(self, stateless_seed=None):
        # does not call super but needs to be type SeedGenerator
        self.stateless_seed = stateless_seed

    def get_config(self):
        return {"stateless_seed": self.stateless_seed}

    def next(self, ordered=None):
        return self.stateless_seed


if keras.backend.backend() == "jax":

    def stateless_to_keras_seed(seed):
        """JAX stateless seeds are already in the correct format for Keras."""
        return seed

else:

    def stateless_to_keras_seed(seed):
        """Converts a stateless seed to a stateless keras SeedGenerator."""
        return StatelessSeedGenerator(seed)


class Distribution:
    def sample(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def log_prob(self, value):
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable()
class GaussianMixture(Distribution):
    def __init__(
        self,
        loc,
        scale,
        mixture_logits=None,
        reparameterize=True,
        reinterpreted_batch_ndims=1,
    ):
        self.loc = loc
        self.scale = scale
        self.mixture_logits = mixture_logits
        self.reparameterize = reparameterize
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def get_config(self):
        return {
            "loc": self.loc,
            "scale": self.scale,
            "mixture_logits": self.mixture_logits,
            "reparameterize": self.reparameterize,
            "reinterpreted_batch_ndims": self.reinterpreted_batch_ndims,
        }

    @classmethod
    def from_config(cls, config):
        config["loc"] = deserialize(config["loc"])
        config["scale"] = deserialize(config["scale"])
        if config["mixture_logits"] is not None:
            config["mixture_logits"] = deserialize(config["mixture_logits"])
        return cls(**config)

    @property
    def required_attributes(self):
        return [
            "loc",
            "scale",
            "mixture_logits",
            "reparameterize",
            "reinterpreted_batch_ndims",
        ]

    @property
    def ready(self):
        return all(hasattr(self, attr) for attr in self.required_attributes)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if self.ready and name in self.required_attributes:
            self.tfd = self.get_tfd()

    def sample(self, *args, seed=None, **kwargs):
        # TODO: implement methods to sample, log_prob, etc. to make tfp redundant
        return self.tfd.sample(*args, **kwargs, seed=seed)

    def log_prob(self, value):
        return self.tfd.log_prob(value)

    def prob(self, value):
        return self.tfd.prob(value)

    def __getattr__(self, name):
        # Prevent recursion
        if name == "tfd":
            raise AttributeError

        # Delegate all other unknown calls to the tfd object
        return getattr(self.tfd, name)

    def get_tfd(self):
        """Converts the distribution to a TensorFlow Probability distribution"""
        # NOTE: https://github.com/tensorflow/probability/issues/1548
        if self.mixture_logits is None:
            return tfd.Independent(
                tfd.MultivariateNormalDiag(self.loc, self.scale),
                reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
            )
        else:
            return tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=self.mixture_logits),
                components_distribution=tfd.Independent(
                    tfd.Normal(
                        loc=self.loc,
                        scale=self.scale,
                    ),
                    reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
                ),
                reparameterize=self.reparameterize,
            )
