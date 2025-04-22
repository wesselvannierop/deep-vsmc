from keras import Sequential, layers


def conv2d_block(
    filters,
    kernel_size=(3, 3),
    padding="same",
    activation="relu",
    norm_layer: layers = None,
):
    block = [layers.Conv2D(filters, kernel_size, padding=padding)]
    if norm_layer is not None:
        block.append(norm_layer())
    if activation is not None:
        block.append(layers.Activation(activation))
    return block


def build_image_encoder(
    img_shape: tuple = (28, 28),
    encoder_depth: int = 3,
    norm_layer: layers = None,
    encoded_dim: int | None = 256,
    nfeatbase: int = 32,
    name: str = None,
    in_channels=1,
    return_layers=False,
):
    # Build encoder
    encoder = [layers.InputLayer((*img_shape, in_channels))]

    for i in range(encoder_depth - 1):
        encoder += conv2d_block(nfeatbase * 2**i, norm_layer=norm_layer)
        encoder.append(layers.MaxPooling2D((2, 2), padding="same"))

    add_flatten_dense = encoded_dim is not None

    encoder += conv2d_block(
        nfeatbase * 2 ** (encoder_depth - 1),
        norm_layer=norm_layer,
        activation="relu" if add_flatten_dense else None,
    )

    if add_flatten_dense:
        encoder.append(layers.MaxPooling2D((2, 2), padding="same"))
        encoder += [layers.Flatten(), layers.Dense(encoded_dim)]

    if return_layers:
        return encoder
    else:
        return Sequential(encoder, name=name)


def build_simple_dense(
    input_shape: tuple,
    output_dim: int,
    depth: int,
    hidden_dim: int = 256,
    activation: str = "relu",
    name: str = None,
):
    nn = [
        layers.InputLayer(input_shape),
        *[layers.Dense(hidden_dim, activation=activation) for _ in range(depth)],
        layers.Dense(output_dim),
    ]
    return Sequential(nn, name=name)
