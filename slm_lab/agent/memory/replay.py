from collections import deque
from slm_lab.agent.memory.base import Memory
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np

logger = logger.get_logger(__name__)


def sample_next_states(head, max_size, ns_idx_offset, batch_idxs, states, ns_buffer):
    """Method to sample next_states from states, with proper guard for next_state idx being out of bound"""
    # idxs for next state is state idxs with offset, modded
    ns_batch_idxs = (batch_idxs + ns_idx_offset) % max_size
    # if head < ns_idx <= head + ns_idx_offset, ns is stored in ns_buffer
    ns_batch_idxs = ns_batch_idxs % max_size
    buffer_ns_locs = np.argwhere(
        (head < ns_batch_idxs) & (ns_batch_idxs <= head + ns_idx_offset)
    ).flatten()
    # find if there is any idxs to get from buffer
    to_replace = buffer_ns_locs.size != 0
    if to_replace:
        # extract the buffer_idxs first for replacement later
        # given head < ns_idx <= head + offset, and valid buffer idx is [0, offset)
        # get 0 < ns_idx - head <= offset, or equiv.
        # get -1 < ns_idx - head - 1 <= offset - 1, i.e.
        # get 0 <= ns_idx - head - 1 < offset, hence:
        buffer_idxs = ns_batch_idxs[buffer_ns_locs] - head - 1
        # set them to 0 first to allow sampling, then replace later with buffer
        ns_batch_idxs[buffer_ns_locs] = 0
    # guard all against overrun idxs from offset
    ns_batch_idxs = ns_batch_idxs % max_size
    next_states = util.batch_get(states, ns_batch_idxs)
    if to_replace:
        # now replace using buffer_idxs and ns_buffer
        buffer_ns = util.batch_get(ns_buffer, buffer_idxs)
        next_states[buffer_ns_locs] = buffer_ns
    return next_states


class Replay(Memory):
    """
    Stores agent experiences and samples from them for agent training

    An experience consists of
        - state: representation of a state
        - action: action taken
        - reward: scalar value
        - next state: representation of next state (should be same as state)
        - done: 0 / 1 representing if the current state is the last in an episode

    The memory has a size of N. When capacity is reached, the oldest experience
    is deleted to make space for the lastest experience.
        - This is implemented as a circular buffer so that inserting experiences are O(1)
        - Each element of an experience is stored as a separate array of size N * element dim

    When a batch of experiences is requested, K experiences are sampled according to a random uniform distribution.

    If 'use_cer', sampling will add the latest experience.

    e.g. memory_spec
    "memory": {
        "name": "Replay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    """

    def __init__(self, memory_spec, agent):
        super().__init__(memory_spec, agent)
        util.set_attr(
            self,
            dict(
                use_cer=False,
            ),
        )
        util.set_attr(
            self,
            self.memory_spec,
            [
                "batch_size",
                "max_size",
                "use_cer",
            ],
        )
        self.max_size = int(
            self.max_size
        )  # convert scientific notation (e.g. 1e6) to int
        self.is_episodic = False
        self.batch_idxs = None
        self.size = 0  # total experiences stored
        self.seen_size = 0  # total experiences seen cumulatively
        self.head = -1  # index of most recent experience
        self.last_sample_head = -1  # head at last sample (for CER)
        # generic next_state buffer to store last next_states (allow for multiple for venv)
        self.ns_idx_offset = agent.env.num_envs if agent.env.is_venv else 1
        self.ns_buffer = deque(maxlen=self.ns_idx_offset)
        # declare what data keys to store
        self.data_keys = [
            "states",
            "actions",
            "rewards",
            "next_states",
            "dones",
            "terminateds",
            "truncateds",
        ]
        self.reset()

    def reset(self):
        """Initializes the memory arrays, size and head pointer"""
        # set self.states, self.actions, ...
        for k in self.data_keys:
            if k != "next_states":  # reuse self.states
                # list add/sample is over 10x faster than np, also simpler to handle
                setattr(self, k, [None] * self.max_size)
        self.size = 0
        self.head = -1
        self.last_sample_head = -1
        self.ns_buffer.clear()

    @lab_api
    def update(self, state, action, reward, next_state, done, terminated, truncated):
        """Interface method to update memory"""
        if self.agent.env.is_venv:
            for s, a, r, ns, d, term, trunc in zip(
                state, action, reward, next_state, done, terminated, truncated
            ):
                self.add_experience(
                    state=s,
                    action=a,
                    reward=r,
                    next_state=ns,
                    done=d,
                    terminated=term,
                    truncated=trunc,
                )
        else:
            self.add_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                terminated=terminated,
                truncated=truncated,
            )

    def add_experience(
        self, *, state, action, reward, next_state, done, terminated, truncated
    ):
        """Implementation for update() to add experience to memory, expanding the memory size if necessary"""
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        # Preserve dtype: uint8 images stay uint8 (memory efficient); everything else float16
        state_dtype = np.uint8 if state.dtype == np.uint8 else np.float16
        self.states[self.head] = state.astype(state_dtype)
        self.actions[self.head] = action
        self.rewards[self.head] = reward
        self.ns_buffer.append(next_state.astype(state_dtype))
        self.dones[self.head] = done
        self.terminateds[self.head] = terminated
        self.truncateds[self.head] = truncated
        # Actually occupied size of memory
        if self.size < self.max_size:
            self.size += 1
        self.seen_size += 1
        # set to_train using memory counters head, seen_size instead of tick since clock will step by num_envs when on venv; to_train will be set to 0 after training step
        algorithm = self.agent.algorithm
        algorithm.to_train = algorithm.to_train or (
            self.seen_size > algorithm.training_start_step
            and self.head % algorithm.training_frequency == 0
        )

    @lab_api
    def sample(self):
        """
        Returns a batch of batch_size samples. Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are an array of the corresponding sampled elements
        e.g.
        batch = {
            'states'     : states,
            'actions'    : actions,
            'rewards'    : rewards,
            'next_states': next_states,
            'dones'      : dones}
        """
        self.batch_idxs = self.sample_idxs(self.batch_size)
        batch = {}
        for k in self.data_keys:
            if k == "next_states":
                batch[k] = sample_next_states(
                    self.head,
                    self.max_size,
                    self.ns_idx_offset,
                    self.batch_idxs,
                    self.states,
                    self.ns_buffer,
                )
            else:
                batch[k] = util.batch_get(getattr(self, k), self.batch_idxs)
        return batch

    def apply_cer(self, batch_idxs):
        """Apply CER: replace some indices with new experiences, return modified indices"""
        num_new_total = (self.head - self.last_sample_head) % self.max_size
        num_new = min(num_new_total, len(batch_idxs), self.size)
        if num_new > 0:
            # Replace last num_new indices with new experiences
            new_idxs = (self.last_sample_head + 1 + np.arange(num_new)) % self.max_size
            batch_idxs = np.concatenate(
                [
                    batch_idxs[:-num_new]
                    if num_new < len(batch_idxs)
                    else np.array([], dtype=int),
                    new_idxs,
                ]
            )
            self.last_sample_head = (self.last_sample_head + num_new) % self.max_size
        return batch_idxs

    def sample_idxs(self, batch_size):
        """Batch indices a sampled random uniformly"""
        batch_idxs = np.random.randint(self.size, size=batch_size)
        if self.use_cer:
            batch_idxs = self.apply_cer(batch_idxs)
        return batch_idxs
