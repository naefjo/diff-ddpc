from abc import ABC, abstractmethod


class BaseController(ABC):
    def __call__(self, obs, *args, **kwargs):
        return self.forward(args, obs, **kwargs)

    @abstractmethod
    def forward(self, obs, *args, **kwargs):
        raise NotImplementedError()
