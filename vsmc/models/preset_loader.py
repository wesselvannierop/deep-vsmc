import importlib.resources
import warnings
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

# Load presets from yaml file
with open(importlib.resources.files("vsmc") / "models" / "presets.yaml") as f:
    presets = yaml.safe_load(f)

models = presets["models"]
names = [model["names"] for model in models]
flat_names = [name for sublist in names for name in sublist]  # Flatten list of lists


def _from_preset(identifier: str, verbose=True) -> Path:
    # Find index and model based on identifier
    index = next((i for i, lst in enumerate(names) if identifier in lst), -1)
    preset = models[index]

    # Print model information
    if verbose:
        print(f"Model: {identifier}")
        if "metadata" in preset["metadata"]:
            discription = preset["metadata"]["description"]
            print(f"Discription: {discription}")

    # Download from huggingface if available
    if "huggingface" in preset:
        try:
            return Path(hf_hub_download(**preset["huggingface"]))
        except Exception as e:
            warnings.warn(f"Failed to download from huggingface: {e}")

    # Load from directory if available
    if "dir" in preset:
        checkpoint = Path(preset["dir"]) / preset["checkpoint"]
        assert checkpoint.exists(), f"Checkpoint {checkpoint} does not exist"
        return checkpoint

    # If neither huggingface or dir is present
    raise ValueError(f"Either huggingface or dir must be present in preset: {preset}")


def from_preset(identifier: str, verbose=True) -> Path:
    """Load a model preset based on the given identifier (or path)"""

    # Check if identifier is a preset
    if isinstance(identifier, str) and identifier.startswith("preset:"):
        identifier = identifier[7:]  # Remove "preset:" prefix
        assert identifier in flat_names, (
            f"Identifier: {identifier} is not a valid preset"
        )
        checkpoint = _from_preset(identifier, verbose)

    # If not a preset, check if it is a path
    elif Path(identifier).exists():
        checkpoint = Path(identifier)

    # If identifier is a dictionary, download from huggingface
    elif isinstance(identifier, dict):
        checkpoint = hf_hub_download(**identifier)

    # If not a preset, path or hf dict, raise error
    else:
        raise ValueError(f"Identifier: {identifier} is not a valid preset or path")

    return checkpoint
