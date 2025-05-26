import abc


class Module(metaclass=abc.ABCMeta):
    """
    Base class for filterflow Modules
    """

    def __init__(self):
        pass

    def __call__(self):
        pass

    @classmethod
    def from_config(cls, config):
        return cls(**config)
