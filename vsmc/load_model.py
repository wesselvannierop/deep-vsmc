import json
import tempfile
import zipfile
from pathlib import Path

import keras
from zea import log


def is_module(obj):
    return isinstance(obj, dict) and "module" in obj


def update_config(json_data, **kwargs):
    config = json_data["config"]
    module = json_data["module"]
    class_name = json_data["class_name"]
    print(f"Updating {module}.{class_name} ...")

    for key, value in kwargs.items():
        # Adding new key
        if key not in config:
            print(f"+ {key}={value}")
            config[key] = value
        # Updating existing key
        elif config[key] != value:
            # If the key is also a module, recursively update the config
            if is_module(config[key]):
                config[key]["config"] = update_config(config[key], **value)
            else:
                print(f"- {key}={config[key]}")
                print(f"+ {key}={value}")
                config[key] = value

    return config


def model_from_json(path, custom_objects=None, **kwargs):
    # Load file
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    content = content.replace("dpf.", "vsmc.")  # rename dpf to vsmc

    # Load json
    json_data = json.loads(content)

    # Update config with **kwargs
    json_data["config"] = update_config(json_data, **kwargs)

    # Load model
    json_string = json.dumps(json_data)
    model = keras.models.model_from_json(json_string, custom_objects)
    return model, json_data


def load_model(checkpoint, custom_objects=None, **kwargs):
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        with zipfile.ZipFile(checkpoint, "r") as zip_ref:
            zip_ref.extractall(checkpoint_dir)
            checkpoint_dir = Path(checkpoint_dir)

            # Log metadata
            metadata_path = checkpoint_dir / "metadata.json"
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.loads(file.read())
                print(f"Metadata: {metadata}")

            # Load model
            json_path = checkpoint_dir / "config.json"
            model, json_data = model_from_json(json_path, custom_objects, **kwargs)
            model_name = json_data["config"]["name"]

            # Build model
            if not model.built:
                model.build_from_config(json_data["build_config"])

            # Load weights
            weights_path = checkpoint_dir / "model.weights.h5"
            model.load_weights(weights_path, skip_mismatch=True)

    log.success(
        f"Succesfully loaded {model_name} with weights from {log.yellow(checkpoint)}"
    )
    return model
