# Modified from: https://github.com/quanvuong/handful-of-trials-pytorch/blob/master/MPC.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.io import savemat

# from DotmapUtils import get_required_argument
from slm_lab.agent.algorithm.optimizer_util import CEMOptimizer
from tqdm import trange
import torch

import numpy as np
from slm_lab.agent.agent import agent_util
from slm_lab.agent.algorithm import meta_algorithm
from slm_lab.agent.algorithm import policy_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api


# net.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# def get_required_argument(dotmap, key, message, default=None):
#     val = dotmap.get(key, default)
#     if val is default:
#         raise ValueError(message)
#     return val


# class Controller:
#     def __init__(self, *args, **kwargs):
#         """Creates class instance.
#         """
#         pass
#
#     def train(self, obs_trajs, acs_trajs, rews_trajs):
#         """Trains this controller using lists of trajectories.
#         """
#         raise NotImplementedError("Must be implemented in subclass.")
#
#     def reset(self):
#         """Resets this controller.
#         """
#         raise NotImplementedError("Must be implemented in subclass.")
#
#     def act(self, obs, t, get_pred_cost=False):
#         """Performs an action.
#         """
#         raise NotImplementedError("Must be implemented in subclass.")
#
#     def dump_logs(self, primary_logdir, iter_logdir):
#         """Dumps logs into primary log directory and per-train iteration log directory.
#         """
#         raise NotImplementedError("Must be implemented in subclass.")


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


class PETS(meta_algorithm.MetaAlgorithm):
    optimizers = {"CEM": CEMOptimizer}

    def __init__(self, agent, global_nets, algorithm_spec,
                 memory_spec, net_spec, algo_idx=0):
        """Creates class instance.
        Arguments:
            params
                .env (gym.env): Environment for which this controller will be used.
                .ac_ub (np.ndarray): (optional) An array of action upper bounds.
                    Defaults to environment action upper bounds.
                .ac_lb (np.ndarray): (optional) An array of action lower bounds.
                    Defaults to environment action lower bounds.
                .per (int): (optional) Determines how often the action sequence will be optimized.
                    Defaults to 1 (reoptimizes at every call to act()).
                .prop_cfg
                    .model_init_cfg (DotMap): A DotMap of initialization parameters for the model.
                        .model_constructor (func): A function which constructs an instance of this
                            model, given model_init_cfg.
                    .model_train_cfg (dict): (optional) A DotMap of training parameters that will be passed
                        into the model every time is is trained. Defaults to an empty dict.
                    .model_pretrained (bool): (optional) If True, assumes that the model
                        has been trained upon construction.
                    .mode (str): Propagation method. Choose between [E, DS, TSinf, TS1, MM].
                        See https://arxiv.org/abs/1805.12114 for details.
                    .npart (int): Number of particles used for DS, TSinf, TS1, and MM propagation methods.
                    .ign_var (bool): (optional) Determines whether or not variance output of the model
                        will be ignored. Defaults to False unless deterministic propagation is being used.
                    .obs_preproc (func): (optional) A function which modifies observations (in a 2D matrix)
                        before they are passed into the model. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc (func): (optional) A function which returns vectors calculated from
                        the previous observations and model predictions, which will then be passed into
                        the provided cost function on observations. Defaults to lambda obs, model_out: model_out.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc2 (func): (optional) A function which takes the vectors returned by
                        obs_postproc and (possibly) modifies it into the predicted observations for the
                        next time step. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .targ_proc (func): (optional) A function which takes current observations and next
                        observations and returns the array of targets (so that the model learns the mapping
                        obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
                        Note: Only needs to process NumPy arrays.
                .opt_cfg
                    .mode (str): Internal optimizer that will be used. Choose between [CEM].
                    .cfg (DotMap): A map of optimizer initializer parameters.
                    .plan_hor (int): The planning horizon that will be used in optimization.
                    .obs_cost_fn (func): A function which computes the cost of every observation
                        in a 2D matrix.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .ac_cost_fn (func): A function which computes the cost of every action
                        in a 2D matrix.
                .log_cfg
                    .save_all_models (bool): (optional) If True, saves models at every iteration.
                        Defaults to False (only most recent model is saved).
                        Warning: Can be very memory-intensive.
                    .log_traj_preds (bool): (optional) If True, saves the mean and variance of predicted
                        particle trajectories. Defaults to False.
                    .log_particles (bool) (optional) If True, saves all predicted particles trajectories.
                        Defaults to False. Note: Takes precedence over log_traj_preds.
                        Warning: Can be very memory-intensive
        """
        super().__init__(agent, global_nets, algorithm_spec, memory_spec, net_spec, algo_idx)
        print(len(self.algorithms))
        # super().__init__(params)

        assert self.agent.body.action_space_is_discrete
        # self.dO, self.dU = params.env.observation_space.shape[0], params.env.action_space.shape[0]
        self.dO, self.dU = self.agent.body.observation_dim, self.agent.body.action_dim
        print("self.dO, self.dU",self.dO, self.dU)
        # self.action_upper_bound, self.action_lower_bound = params.env.action_space.high, params.env.action_space.low
        # self.action_upper_bound = np.minimum(self.action_upper_bound, params.get("ac_ub", self.action_upper_bound))
        # self.action_lower_bound = np.maximum(self.action_lower_bound, params.get("ac_lb", self.action_lower_bound))
        self.action_upper_bound, self.action_lower_bound = np.array(self.agent.body.action_dim), np.array([0])
        self.update_fns = [] # params.get("update_fns", [])
        self.per = 1 # params.get("per", 1)


        self.ensemble_of_n = 2 # 5 # self.model_init_cig = params.prop_cfg.get("model_init_cfg", {})
        # self.model_train_cfg = params.prop_cfg.get("model_train_cfg", {})
        self.fit_during_n_epochs = 2 # 5
        # self.prop_mode = get_required_argument(params.prop_cfg, "mode", "Must provide propagation method.")
        self.npart = 10 # 20 # get_required_argument(params.prop_cfg, "npart", "Must provide number of particles.")
        self.ign_var = False # params.prop_cfg.get("ign_var", False) or self.prop_mode == "E"

        self.obs_preproc = (lambda obs: obs) # params.prop_cfg.get("obs_preproc", lambda obs: obs)
        self.obs_postproc = (lambda obs, model_out: model_out) # params.prop_cfg.get("obs_postproc", lambda obs, model_out: model_out)
        self.obs_postproc2 = (lambda next_obs: next_obs) # params.prop_cfg.get("obs_postproc2", lambda next_obs: next_obs)
        self.targ_proc = (lambda obs, next_obs: next_obs) # params.prop_cfg.get("targ_proc", lambda obs, next_obs: next_obs)

        self.opt_mode = 'CEM' #get_required_argument(params.opt_cfg, "mode", "Must provide optimization method.")
        self.planning_horizon = 2 # get_required_argument(params.opt_cfg, "plan_hor", "Must provide planning horizon.")
        # self.obs_cost_fn =  get_required_argument(params.opt_cfg, "obs_cost_fn", "Must provide cost on observations.")
        # self.ac_cost_fn = (lambda ac : 0) # get_required_argument(params.opt_cfg, "ac_cost_fn",
        #                                                         # "Must provide cost on actions.")

        self.save_all_models = False # params.log_cfg.get("save_all_models", False)
        self.log_traj_preds = False # params.log_cfg.get("log_traj_preds", False)
        self.log_particles = False # params.log_cfg.get("log_particles", False)

        # Perform argument checks
        assert self.opt_mode == 'CEM'
        # assert self.prop_mode == 'TSinf', 'only TSinf propagation mode is supported'
        # assert self.npart % self.model_init_cig.num_nets == 0, "Number of particles must be a multiple of the ensemble size."
        assert self.npart % self.ensemble_of_n == 0, "Number of particles must be a multiple of the ensemble size."

        # Create action sequence optimizer
        # opt_cfg = params.opt_cfg.get("cfg", {})
        opt_cfg = {
                    "Random": {
                        "popsize": 2000
                    },
                    "CEM": {
                        "popsize": 400,
                        "num_elites": 40,
                        "max_iters": 5,
                        "alpha": 0.1
                    }
                }
        self.optimizer = CEMOptimizer(
            sol_dim=self.planning_horizon * self.dU,
            lower_bound=np.tile(self.action_lower_bound, [self.planning_horizon]),
            upper_bound=np.tile(self.action_upper_bound, [self.planning_horizon]),
            cost_function=self._compile_cost,
            **opt_cfg['CEM']
        )

        # Controller state variables
        self.has_been_trained = False #params.prop_cfg.get("model_pretrained", False)
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.action_lower_bound + self.action_upper_bound) / 2, [self.planning_horizon])
        self.init_var = np.tile(np.square(self.action_upper_bound - self.action_lower_bound) / 16, [self.planning_horizon])
        # self.train_in = np.array([]).reshape(0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
        # self.train_targs = np.array([]).reshape(
        #     0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1]
        # )

        # print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart) +
        #       ("Ignoring variance." if self.ign_var else ""))

        if self.save_all_models:
            print("Controller will save all models. (Note: This may be memory-intensive.")
        if self.log_particles:
            print("Controller is logging particle predictions (Note: This may be memory-intensive).")
            self.pred_particles = []
        elif self.log_traj_preds:
            print("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
        else:
            print("Trajectory prediction logging is disabled.")

        # # Set up pytorch model
        # self.model = get_required_argument(
        #     params.prop_cfg.model_init_cfg, "model_constructor", "Must provide a model constructor."
        # )(params.prop_cfg.model_init_cfg)

        self.device = self.algorithms[0].net.device

        if len(self.agent.body.observation_dim) == 1 :
            self.max_logvar = torch.nn.Parameter(torch.ones(1, self.agent.body.observation_dim[0] // 2,
                                                            dtype=torch.float32) / 2.0)
            self.min_logvar = torch.nn.Parameter(- torch.ones(1, self.agent.body.observation_dim[0] // 2,
                                                              dtype=torch.float32) * 10.0)
        else:
            raise NotImplementedError()

        for algo in self.algorithms:
            algo.meta_algo = self




    @lab_api
    def act(self, state):
        """Returns the action that this controller would take at time t given observation obs.
        Arguments:
            state: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.
        Returns: An action (and possibly the predicted cost)
        """
        if not self.has_been_trained:
            return np.random.uniform(self.action_lower_bound, self.action_upper_bound, self.action_lower_bound.shape)
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]

            self.reset()
            return action

        self.sy_cur_obs = state

        soln = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dU:], np.zeros(self.per * self.dU)])
        self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)

        return self.act(state)

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        raise NotImplementedError()
        # return self.algorithms[self.active_algo_idx].sample()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def training_frequency(self):
        raise NotImplementedError()

    @property
    def net(self):
        raise NotImplementedError()
        # return self.algorithms[self.active_algo_idx].net

    # @lab_api
    # def train(self):
    #     '''Implement algorithm train, or throw NotImplementedError'''
    #     losses = []
    #
    #     for idx, algo in enumerate(self.algorithms):
    #         if self.agent.world.deterministic:
    #             self.agent.world._set_rd_state(self.agent.world.rd_seed)
    #         losses.append(algo.train())
    #
    #     # TODO clip_eps_scheduler
    #
    #     losses = [el for el in losses if not np.isnan(el)]
    #     loss = sum(losses) if len(losses) > 0 else np.nan
    #
    #     if not np.isnan(loss):
    #         logger.debug(f"{self.active_algo_idx} loss {loss}")
    #
    #     return loss

    @lab_api
    def update(self):
        '''Implement algorithm update, or throw NotImplementedError'''
        explore_vars = []
        for algo in self.algorithms:
            explore_vars.append(algo.update())
        explore_vars = [el for el in explore_vars if not np.isnan(el)]
        explore_var = sum(explore_vars) if len(explore_vars) > 0 else np.nan
        return explore_var

    @lab_api
    def memory_update(self, state, action, welfare, next_state, done):
        return self.memory.update(state, action, welfare, next_state, done)
        # return self.algorithms[self.active_algo_idx].memory_update(state, action, welfare, next_state, done)

    @lab_api
    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.
        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.
            rews_trajs: A list of reward arrays.
        Returns: None.
        """

        # Construct new training points and add to training set
        # new_train_in, new_train_targs = [], []
        # for obs, acs in zip(obs_trajs, acs_trajs):
        #     new_train_in.append(np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1))
        #     new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))
        # self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
        # self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

        # Train the model
        self.has_been_trained = True

        # Train the pytorch model
        self.model.fit_input_stats(self.train_in)

        # idxs = np.random.randint(self.train_in.shape[0], size=[self.model.num_nets, self.train_in.shape[0]])

        # epochs = self.fit_during_n_epochs # self.model_train_cfg['epochs']

        # TODO: double-check the batch_size for all env is the same
        batch_size = 32

        for bayesian_algo in self.algorithms[1:]:
            # epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
            # num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

            # for _ in epoch_range:
            for _ in range(self.fit_during_n_epochs):
                for step_idx, data in enumerate(self.memory.replay_all_history()):

                    s = data["states"]
                    a = data["actions"]
                    w = data["rewards"]
                    n_s = data["next_states"]
                    done = data["dones"]

                    bayesian_algo.memory_update(s, a, float(w), n_s, bool(done))

                    if bayesian_algo.to_train == 1:

                        bayesian_algo.train()
                        bayesian_algo.update()

                        # batch_idxs = idxs[:, batch_num * batch_size: (batch_num + 1) * batch_size]

                        # TODO create the loss function
                        # loss = 0.01 * (self.model.max_logvar.sum() - self.model.min_logvar.sum())
                        # loss += self.model.compute_decays()
                        #
                        # # TODO: move all training data to GPU before hand
                        # train_in = torch.from_numpy(self.train_in[batch_idxs]).to(net.device).float()
                        # train_targ = torch.from_numpy(self.train_targs[batch_idxs]).to(net.device).float()
                        #
                        # mean, logvar = self.model(train_in, ret_logvar=True)
                        # inv_var = torch.exp(-logvar)
                        #
                        # train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
                        # train_losses = train_losses.mean(-1).mean(-1).sum()
                        # # Only taking mean over the last 2 dimensions
                        # # The first dimension corresponds to each model in the ensemble
                        #
                        # loss += train_losses
                        #
                        # self.model.optim.zero_grad()
                        # loss.backward()
                        # self.model.optim.step()

            # idxs = shuffle_rows(idxs)

            # val_in = torch.from_numpy(self.train_in[idxs[:5000]]).to(net.device).float()
            # val_targ = torch.from_numpy(self.train_targs[idxs[:5000]]).to(net.device).float()
            #
            # mean, _ = self.model(val_in)
            # mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)
            #
            # epoch_range.set_postfix({
            #     "Training loss(es)": mse_losses.detach().cpu().numpy()
            # })

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).
        Returns: None
        """
        self.prev_sol = np.tile((self.action_lower_bound + self.action_upper_bound) / 2, [self.planning_horizon])
        self.optimizer.reset()

        # for update_fn in self.update_fns:
        #     update_fn()

    # def act(self, obs, t, get_pred_cost=False):
    #     """Returns the action that this controller would take at time t given observation obs.
    #     Arguments:
    #         obs: The current observation
    #         t: The current timestep
    #         get_pred_cost: If True, returns the predicted cost for the action sequence found by
    #             the internal optimizer.
    #     Returns: An action (and possibly the predicted cost)
    #     """
    #     if not self.has_been_trained:
    #         return np.random.uniform(self.action_lower_bound, self.action_upper_bound, self.action_lower_bound.shape)
    #     if self.ac_buf.shape[0] > 0:
    #         action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
    #         return action
    #
    #     self.sy_cur_obs = obs
    #
    #     soln = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
    #     self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dU:], np.zeros(self.per * self.dU)])
    #     self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)
    #
    #     return self.act(obs, t)

    # def dump_logs(self, primary_logdir, iter_logdir):
    #     """Saves logs to either a primary log directory or another iteration-specific directory.
    #     See __init__ documentation to see what is being logged.
    #     Arguments:
    #         primary_logdir (str): A directory path. This controller assumes that this directory
    #             does not change every iteration.
    #         iter_logdir (str): A directory path. This controller assumes that this directory
    #             changes every time dump_logs is called.
    #     Returns: None
    #     """
    #     # TODO: implement saving model for pytorch
    #     # self.model.save(iter_logdir if self.save_all_models else primary_logdir)
    #     if self.log_particles:
    #         savemat(os.path.join(iter_logdir, "predictions.mat"), {"predictions": self.pred_particles})
    #         self.pred_particles = []
    #     elif self.log_traj_preds:
    #         savemat(
    #             os.path.join(iter_logdir, "predictions.mat"),
    #             {"means": self.pred_means, "vars": self.pred_vars}
    #         )
    #         self.pred_means, self.pred_vars = [], []

    @torch.no_grad()
    def _compile_cost(self, ac_seqs):

        nopt = ac_seqs.shape[0]

        ac_seqs = torch.from_numpy(ac_seqs).float().to(self.device)

        # Reshape ac_seqs so that it's amenable to parallel compute
        # Before, ac seqs has dimension (400, 25) which are pop size and sol dim coming from CEM
        ac_seqs = ac_seqs.view(-1, self.planning_horizon, self.dU)
        #  After, ac seqs has dimension (400, 25, 1)

        transposed = ac_seqs.transpose(0, 1)
        # Then, (25, 400, 1)

        expanded = transposed[:, :, None]
        # Then, (25, 400, 1, 1)

        tiled = expanded.expand(-1, -1, self.npart, -1)
        # Then, (25, 400, 20, 1)

        ac_seqs = tiled.contiguous().view(self.planning_horizon, -1, self.dU)
        # Then, (25, 8000, 1)

        # Expand current observation
        cur_obs = torch.from_numpy(self.sy_cur_obs).float().to(self.device)
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(nopt * self.npart, -1)

        costs = torch.zeros(nopt, self.npart, device=self.device)

        for t in range(self.planning_horizon):
            cur_acs = ac_seqs[t]

            cost = []
            for bayesian_algo in self.algorithms[1:]:
                next_obs = self._predict_next_obs(bayesian_algo, cur_obs, cur_acs)
                cost.appedn(self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs))
            cost = torch.stack(cost, dim=-1)

            cost = cost.view(-1, self.npart)

            costs += cost
            cur_obs = self.obs_postproc2(next_obs)

        # Replace nan with high cost
        costs[costs != costs] = 1e6

        return costs.mean(dim=1).detach().cpu().numpy()

    def _predict_next_obs(self, one_algo, obs, acs):
        proc_obs = self.obs_preproc(obs)

        assert self.prop_mode == 'TSinf'

        proc_obs = self._expand_to_ts_format(proc_obs)
        acs = self._expand_to_ts_format(acs)

        inputs = torch.cat((proc_obs, acs), dim=-1)

        # mean, var = one_algo.net(inputs)
        pdparam = one_algo.proba_distrib_params(inputs)
        half_len = int(len(pdparam) / 2)
        mean, logvar = pdparam[:half_len], pdparam[half_len:]
        var = torch.exp(logvar)

        predictions = mean + torch.randn_like(mean, device=one_algo.net.device) * var.sqrt()

        # TS Optimization: Remove additional dimension
        predictions = self._flatten_to_matrix(predictions)

        return self.obs_postproc(obs, predictions)

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        # Before, [10, 5] in case of proc_obs
        reshaped = mat.view(-1, self.model.num_nets, self.npart // self.model.num_nets, dim)
        # After, [2, 5, 1, 5]

        transposed = reshaped.transpose(0, 1)
        # After, [5, 2, 1, 5]

        reshaped = transposed.contiguous().view(self.model.num_nets, -1, dim)
        # After. [5, 2, 5]

        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]

        reshaped = ts_fmt_arr.view(self.model.num_nets, -1, self.npart // self.model.num_nets, dim)

        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(-1, dim)

        return reshaped