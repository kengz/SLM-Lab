# from abc import ABC, abstractmethod
#
# #TODO make it really useful (not useless)
# class ObservableAgentInterface(object):
#     _welfare = None
#
#     @property
#     @abstractmethod
#     def reward(self):
#         raise NotImplementedError
#
#     @property
#     @abstractmethod
#     def action(self):
#         raise NotImplementedError
#
#     @property
#     @abstractmethod
#     def state(self):
#         raise NotImplementedError
#
#     @property
#     @abstractmethod
#     def welfare(self):
#         return self._welfare
#
#     @welfare.setter
#     def welfare(self, value):
#         self._welfare = value
#
#     @property
#     @abstractmethod
#     def next_state(self):
#         raise NotImplementedError
#
#     @property
#     @abstractmethod
#     def done(self):
#         raise NotImplementedError
#
#     @property
#     @abstractmethod
#     def algorithm(self):
#         raise NotImplementedError
#
