import os

backend = os.environ.get("KERAS_BACKEND")

if backend == "tensorflow":
    from .state import StateSeriesTF as StateSeries
    from .state import StateTF as State
    from .state import write_state_array
else:
    from .state import State, StateSeries
    from .state import write_state_array_jax as write_state_array
