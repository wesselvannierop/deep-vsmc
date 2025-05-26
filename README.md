# Deep Variational Sequential Monte Carlo for High-Dimensional Observations

Paper: [[ArXiv](https://arxiv.org/abs/2501.05982)]; [[ICASSP 2025](https://doi.org/10.1109/ICASSP49660.2025.10889044)]
by [Wessel L. van Nierop](https://wesselvannierop.com), Nir Schlezinger, and Ruud J.G. van Sloun

## Usage

### Installation through pip

```bash
pip install -e .
```

### Installation through docker

```bash
docker pull wessel2105/vsmc:latest
```

Or use the included `Dockerfile` to build your own image:

```bash
docker build -t vsmc:latest .
```

### Example usage

```bash
python run.py
```

## Credits

A large part of this codebase is based on [filterflow](https://github.com/JTT94/filterflow).
We have extended that codebase to also work with jax through [keras 3.x](https://keras.io/keras_3/).