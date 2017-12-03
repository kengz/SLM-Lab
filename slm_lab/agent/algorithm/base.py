from abc import ABC, abstractmethod, abstractproperty


class Algorithm(ABC):
    '''
    Abstract class ancestor to all Algorithms,
    specifies the necessary design blueprint for agent to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''

    def __init__(self, agent):
        self.agent = agent

    @abstractmethod
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        raise NotImplementedError

    def body_act_discrete(self, body, body_state):
        '''Implement atomic discrete body_action, or throw NotImplementedError. E.g. fetch body_action from net given body info.'''
        raise NotImplementedError
        return body_action

    def body_act_continuous(self, body, body_state):
        '''Implement atomic continuous body_action, or throw NotImplementedError. E.g. fetch body_action from net given body info.'''
        raise NotImplementedError
        return body_action

    def body_act(self, body, body_state):
        '''Standard atomic body_act method. Atomic body logic should be implemented in submethods.'''
        if body.is_discrete:
            return self.body_act_discrete(body, body_state)
        else:
            return self.body_act_continuous(body, body_state)

    def act(self, state):
        '''Interface-level agent act method for all its bodies. Resolves state to body_state; get body_action and compose into action.'''
        action = []
        for eb_idx, body in enumerate(self.agent.bodies):
            body_state = state[eb_idx]
            body_action = self.body_act(body, body_state)
            action.append(body_action)
        return action

    @abstractmethod
    def train(self):
        '''Implement algorithm train, or throw NotImplementedError'''
        raise NotImplementedError

    @abstractmethod
    def update(self):
        '''Implement algorithm update, or throw NotImplementedError'''
        raise NotImplementedError
