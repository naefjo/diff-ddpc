from abc import ABC, abstractmethod


class BaseController(ABC):
    def __call__(self, state, reference, *args, **kwargs):
        return self.forward(args, state, reference, **kwargs)

    @abstractmethod
    def forward(self, state, reference, *args, **kwargs):
        raise NotImplementedError()
