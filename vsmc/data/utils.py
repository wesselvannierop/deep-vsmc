def backwards_compatibility(config):
    if "image_shape" not in config:
        assert "img_size" in config, "image_shape or img_size must be in the config"
        img_size = config.pop("img_size")
        config["image_shape"] = (img_size, img_size)
    if "likelihood_sigma" in config:
        config["sigma"] = config.pop("likelihood_sigma")
    return config
