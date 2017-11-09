'''
The agent module
Contains graduated components from experiments for building agents
and be taught, tested, evaluated on curriculum.
To be designed by human and evolution module,
based on the experiment aim (trait) and fitness metrics.
Main SLM components (refer to SLM doc for more):
- primary survival objective
- control policies
- sensors (input) for embodiment
- motors (output) for embodiment
- neural architecture
- memory (with time)
- prioritization mechanism and "emotions"
- strange loop must be created
- social aspect
- high level properties of thinking, e.g. creativity, planning.

Agent components:
- algorithms
- memory
- net
- policy
'''

from abc import ABC, abstractmethod, abstractproperty


class Agent(ABC):
    # eventually, wanna init with string name of components
    # but dont wanna be too restrictive
    # start with a random agent first, with default components
    # tie into a trial of experiment (every run is a trial)
    @abstractproperty
    def name(self):
        return 'name'

    @abstractmethod
    def foo(self):
        pass
