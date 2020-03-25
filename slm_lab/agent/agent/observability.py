from abc import ABC, abstractmethod

class ObservableAgentInterface(object):

    @property
    @abstractmethod
    def reward(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def action(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def welfare(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def next_state(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def done(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def algorithm(self):
        raise NotImplementedError