import os


def runnable(
    backend="tensorflow",
    device="auto:1",
    hide_first_for_tf=None,  # will use the first GPU for JAX and the second GPU for TensorFlow.
    matplotlib_backend="agg",  # for running on server
):
    if bool(os.environ.get("RAN_RUNNABLE", False)):
        return

    if hide_first_for_tf is None:
        hide_first_for_tf = backend == "tensorflow" and device == "auto:2"

    # Set the backend
    os.environ["KERAS_BACKEND"] = backend
    print(f"[dpf]: Using {backend} backend 🔥!")

    # Set CUDA_VISIBLE_DEVICES
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        import zea

        zea.init_device(device, allow_preallocate=False)

        if hide_first_for_tf:
            import tensorflow as tf

            visible_devices = tf.config.get_visible_devices()
            tf.config.set_visible_devices([visible_devices[0], visible_devices[-1]])

    # Set the matplotlib backend
    if matplotlib_backend is not None:
        import matplotlib.pyplot as plt

        plt.switch_backend(matplotlib_backend)  # for running on server

    os.environ["RAN_RUNNABLE"] = "True"
