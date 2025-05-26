import keras
from keras import Sequential, layers, ops
from keras_cv.src.models.stable_diffusion.padded_conv2d import PaddedConv2D


class AttentionBlock(layers.Layer):
    """
    Copied from keras_cv.src.models.stable_diffusion.attention_block.AttentionBlock
    Adds `groups` parameter to the original implementation.
    """

    def __init__(self, output_dim, groups=32, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.groups = groups

        self.norm = layers.GroupNormalization(epsilon=1e-5, groups=groups)
        self.q = PaddedConv2D(output_dim, 1)
        self.k = PaddedConv2D(output_dim, 1)
        self.v = PaddedConv2D(output_dim, 1)
        self.proj_out = PaddedConv2D(output_dim, 1)

    def call(self, inputs):
        x = self.norm(inputs)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # Compute attention
        shape = ops.shape(q)
        h, w, c = shape[1], shape[2], shape[3]
        q = ops.reshape(q, (-1, h * w, c))  # b, hw, c
        k = ops.transpose(k, (0, 3, 1, 2))
        k = ops.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * 1 / ops.sqrt(ops.cast(c, self.compute_dtype))
        y = keras.activations.softmax(y)

        # Attend to values
        v = ops.transpose(v, (0, 3, 1, 2))
        v = ops.reshape(v, (-1, c, h * w))
        y = ops.transpose(y, (0, 2, 1))
        x = v @ y
        x = ops.transpose(x, (0, 2, 1))
        x = ops.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs


class ResnetBlock(layers.Layer):
    """
    Copied from: https://github.com/keras-team/keras-cv/blob/master/keras_cv/src/models/stable_diffusion/resnet_block.py#L19
    Adds `groups` parameter to the original implementation.
    """

    def __init__(self, output_dim, groups=32, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.groups = groups

        self.norm1 = layers.GroupNormalization(epsilon=1e-5, groups=groups)
        self.conv1 = PaddedConv2D(output_dim, 3, padding=1)
        self.norm2 = layers.GroupNormalization(epsilon=1e-5, groups=groups)
        self.conv2 = PaddedConv2D(output_dim, 3, padding=1)

    def build(self, input_shape):
        if input_shape[-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        x = self.conv1(keras.activations.swish(self.norm1(inputs)))
        x = self.conv2(keras.activations.swish(self.norm2(x)))
        return x + self.residual_projection(inputs)


def get_image_encoder_model(
    in_channels: int = 1,
    out_channels: int = 4,
    name="image_encoder",
    return_layers=False,
    nfeatbase=128,
):
    """
    Inspired by the encoder of [stable diffusion](https://github.com/keras-team/keras-cv/blob/master/keras_cv/src/models/stable_diffusion/image_encoder.py)
    In contrast to the original implementation, you can change `nfeatbase`
    to control the number of features in the first layer. Also, the `in_channels` parameter can
    be changed.
    """
    N_LAYERS = 3  # fixed
    features = [nfeatbase * 2**i for i in range(N_LAYERS)]
    groups = min(*features, 32)

    def block(nfeat):
        return [
            ResnetBlock(nfeat, groups=groups),
            ResnetBlock(nfeat, groups=groups),
            PaddedConv2D(nfeat, 3, padding=((0, 1), (0, 1)), strides=2),
        ]

    encoder = [
        layers.InputLayer((None, None, in_channels)),
        PaddedConv2D(features[0], 3, padding=1),
        *block(features[0]),
        *block(features[1]),
        *block(features[2]),
        ResnetBlock(features[2], groups=groups),
        ResnetBlock(features[2], groups=groups),
        ResnetBlock(features[2], groups=groups),
        AttentionBlock(features[2], groups=groups),
        ResnetBlock(features[2], groups=groups),
        layers.GroupNormalization(epsilon=1e-5, groups=groups),
        layers.Activation("swish"),
        PaddedConv2D(out_channels, 3, padding=1),
        PaddedConv2D(out_channels, 1),
    ]

    if return_layers:
        return encoder
    else:
        return Sequential(encoder, name=name)


def get_proposal_model(
    in_channels: int = 8,
    out_channels: int = 4,
    return_layers=False,
    name="proposal",
    nfeat=128,
):
    # NOTE: this NN does not change img_shape
    groups = min(nfeat, 32)
    proposal = [
        layers.InputLayer((None, None, in_channels)),
        PaddedConv2D(nfeat, 3, padding=1),
        #
        ResnetBlock(nfeat, groups=groups),
        ResnetBlock(nfeat, groups=groups),
        ResnetBlock(nfeat, groups=groups),
        AttentionBlock(nfeat, groups=groups),
        ResnetBlock(nfeat, groups=groups),
        layers.GroupNormalization(epsilon=1e-5, groups=groups),
        layers.Activation("swish"),
        PaddedConv2D(out_channels, 3, padding=1),
        PaddedConv2D(out_channels, 1),
    ]

    if return_layers:
        return proposal
    else:
        return Sequential(proposal, name=name)


if __name__ == "__main__":
    get_image_encoder_model(nfeatbase=16).summary()

    mixture = 1
    latent_channels = 4
    in_channels = latent_channels * 2
    out_channels = 2 * latent_channels * mixture
    equally_weighted_mixture = False
    if not equally_weighted_mixture:
        out_channels += mixture
    get_proposal_model(in_channels, out_channels).summary()
