import tensorflow as tf
from keras import ops

import tfp_wrapper as tfp

tfd = tfp.distributions
kl = tfd.kullback_leibler

NUM_MC_SAMPLES = 1000
BATCH_SIZE = 30  # TODO: adapt to VRAM


# https://github.com/tensorflow/probability/issues/199
# https://colab.research.google.com/drive/1RHx23ViJL-1SIgVYbUjB8QjCeQ9b2fVS
@kl.RegisterKL(tfd.MixtureSameFamily, tfd.MixtureSameFamily)
def _mc_kl_msf_msf(a, b, seed=None, name="_mc_kl_msf_msf"):
    with tf.name_scope(name):
        return mc_kld(a, b, seed)


def entropy_mc(dist, n_samples=NUM_MC_SAMPLES, batch_size=BATCH_SIZE):
    """
    Calculates the Monte Carlo estimate of entropy for a given probability distribution.
    # TODO: add seed support

    Args:
        dist (tfp.distributions.Distribution): The probability distribution.
        n_samples (int, optional): The number of samples to draw. Defaults to 100.
        batch_size (int, optional): The batch size for drawing samples. Defaults to BATCH_SIZE.
    """

    partial_sum = ops.zeros_like(dist.log_prob(dist.sample()))
    for _ in ops.arange(n_samples // batch_size):
        samples = dist.sample(batch_size)
        partial_sum += ops.sum(dist.log_prob(samples), axis=0)
    return -partial_sum / n_samples


def mc_kld(a, b, n_samples=NUM_MC_SAMPLES, batch_size=BATCH_SIZE):
    """
    Monte Carlo estimation of the Kullback-Leibler divergence.
    # TODO: add seed support

    Args:
        a: distribution A, must have a sample method and a log_prob method
        b: distribution B, must have a log_prob method
        n_samples: int
        batch_size: int
    """
    partial_sum = ops.zeros_like(a.log_prob(a.sample()))
    for _ in range(n_samples // batch_size):
        s = a.sample(batch_size)
        partial_sum += ops.sum(a.log_prob(s) - b.log_prob(s), axis=0)
    return partial_sum / n_samples


class MultivariateNormalFullCovariance(tfd.MultivariateNormalTriL):
    def __init__(self, mean, covariance):
        super().__init__(mean, ops.cholesky(covariance))


if __name__ == "__main__":
    # Example:
    locs = tf.constant([[-1.0], [1.0]])
    locs2 = tf.constant([[-1.0], [1.0]])

    with tf.GradientTape() as tape:
        a = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
            components_distribution=tfd.MultivariateNormalDiag(
                locs, tf.ones_like(locs)
            ),
        )

        b = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.1, 0.9]),
            components_distribution=tfd.MultivariateNormalDiag(
                locs2, tf.ones_like(locs2)
            ),
        )

        tape.watch(locs)
        kl_ab = a.kl_divergence(b)
        grad = tape.gradient(kl_ab, locs)
    print(kl_ab)
    print(grad)
