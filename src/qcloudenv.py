# qcloud_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from des import DES, generatingJob
from flowScheduler import  flow_scheduler_1
from jobScheduler import job_scheduler
from cluster import qCloud, create_random_topology
import random
import time
from jobScheduler import placement
import math


class QCloudBatchEnv(gym.Env):
    """
    Batch RL environment for QPU selection (multi-binary actions).

    - Each episode: a batch of jobs.
    - Each step: pick a subset of QPUs (multi-binary vector) to run the current job.
    - Reward:
        - Negative communication cost for the current job.
        - Penalty for invalid actions / unscheduled jobs.
        - Final penalty for total unscheduled jobs at end of episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        qcloud,
        job_generator,
        jobs_per_episode=8,
        reward_mode="simple",
        invalid_penalty=-5.0,
        unscheduled_penalty=-2.0,
        comm_cost_scale=1.0,
    ):
        super().__init__()
        assert reward_mode in ["simple", "wig", "execution"]

        self.qcloud = qcloud
        self.job_generator = job_generator
        self.jobs_per_episode = jobs_per_episode
        self.reward_mode = reward_mode
        self.invalid_penalty = float(invalid_penalty)
        self.unscheduled_penalty = float(unscheduled_penalty)
        self.scheduler = job_scheduler(None, None, self.qcloud, scheduler_type='default')
        self.des = None
        self.flow_scheduler = None
        if self.reward_mode == "execution":
          #create a new logger so i can grab execution times
          self.des = DES(self.qcloud, self.scheduler, None)
          self.flow_scheduler = flow_scheduler_1(self.des, self.qcloud, epr_p=0.3, name="BFS")

        self.num_qpus = len(self.qcloud.qpus)
        self.total_qubits = np.array(
            [q.ncp_qubits for q in self.qcloud.qpus],
            dtype=np.float32,
        )

        # features
        self.node_feature_dim = 3
        self.job_feature_dim = 3

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_qpus, self.node_feature_dim),
                    dtype=np.float32,
                ),
                "job_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.job_feature_dim,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_qpus,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.MultiBinary(self.num_qpus)

        # episode state
        self.jobs = None
        self.current_job_idx = None
        self.current_job = None
        self.qpu_available = None
        self.unscheduled_jobs = None
        self.step_count = None

        self.degrees = np.array(
            [self.qcloud.network.degree(i) for i in range(self.num_qpus)],
            dtype=np.float32,
        )
        self.max_degree = max(1.0, float(self.degrees.max()))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #change this also try for random job creation
        self.jobs = self.job_generator.generate_job(self.jobs_per_episode, time_frame=10, step=1, probability=[0.2, 0.3, 0.5])
        self.current_job_idx = 0
        self.current_job = self.jobs[0]

        self.qpu_available = np.array(
            [q.available_qubits for q in self.qcloud.qpus],
            dtype=np.float32,
        )

        self.unscheduled_jobs = 0
        self.step_count = 0

        obs = self._build_observation()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1
        action = np.array(action, dtype=np.int32)
        selected_qpus = np.where(action == 1)[0].tolist()

        done = False
        truncated = False
        info = {}
        reward = 0.0
        invalid = False

        job = self.current_job
        job_nq, job_depth, job_n2 = self._extract_job_stats(job)

        # validate action
        if len(selected_qpus) == 0:
            invalid = True

        total_sel_qubits = (
            float(self.qpu_available[selected_qpus].sum())
            if not invalid
            else 0.0
        )
        if not invalid and total_sel_qubits < job_nq:
            invalid = True

        if invalid:
            reward += self.invalid_penalty
            self.unscheduled_jobs += 1
        else:
            remaining = job_nq
            q_sorted = sorted(
                selected_qpus,
                key=lambda idx: self.qpu_available[idx],
                reverse=True,
            )
            for qi in q_sorted:
                if remaining <= 0:
                    break
                can_take = min(remaining, self.qpu_available[qi])
                self.qpu_available[qi] -= can_take
                remaining -= can_take

            if remaining > 0:
                reward += self.invalid_penalty
                self.unscheduled_jobs += 1
            else:
              circuit = job.circuit
              wig = self.scheduler.convert_to_weighted_graph(circuit)
              if len(selected_qpus) == 1:
                place = placement(job, 1, [self.qcloud.qpus[i] for i in selected_qpus], wig)
              else:
                place = placement(job, self.scheduler.partition_circuit(len(selected_qpus),wig), [self.qcloud.qpus[i] for i in selected_qpus], wig)

              if self.reward_mode == "simple":
                comm_cost = place.get_communication_cost()
                reward -= comm_cost
              else:
                #add job to queue for flow scheduler
                place.start_time = self.des.current_time
                self.scheduler._schedule_new(place, job, "BFS")
                self.scheduler.scheduled_job.append(job)

        self.current_job_idx += 1
        if self.current_job_idx >= len(self.jobs):
            done = True
        else:
            self.current_job = self.jobs[self.current_job_idx]

        if done:
            if self.reward_mode == "execution":
              self.flow_scheduler.run()
              self.des.run()
            reward -= self.unscheduled_penalty * float(self.unscheduled_jobs)
            info["unscheduled_jobs"] = int(self.unscheduled_jobs)

        obs = (
            self._build_observation()
            if not done
            else self._build_terminal_observation()
        )
        return obs, float(reward), done, truncated, info

    def _build_observation(self):
        total_qubits_norm = self.total_qubits / np.maximum(
            self.total_qubits.max(), 1.0
        )
        available_norm = self.qpu_available / np.maximum(self.total_qubits, 1.0)
        degree_norm = self.degrees / self.max_degree

        node_features = np.stack(
            [available_norm, total_qubits_norm, degree_norm], axis=-1
        ).astype(np.float32)

        job = self.current_job
        n_qubits, depth, n_2qb = self._extract_job_stats(job)

        rel_qubits = n_qubits / (self.total_qubits.sum() + 1e-6)
        rel_depth = depth / (depth + 1.0)
        density = n_2qb / max(n_qubits, 1.0)
        density = density / (density + 1.0)

        job_features = np.array(
            [rel_qubits, rel_depth, density], dtype=np.float32
        )

        action_mask = (self.qpu_available > 0).astype(np.float32)

        return {
            "node_features": node_features,
            "job_features": job_features,
            "action_mask": action_mask,
        }

    def _build_terminal_observation(self):
        return {
            "node_features": np.zeros(
                (self.num_qpus, self.node_feature_dim), dtype=np.float32
            ),
            "job_features": np.zeros(
                (self.job_feature_dim,), dtype=np.float32
            ),
            "action_mask": np.zeros((self.num_qpus,), dtype=np.float32),
        }

    def _extract_job_stats(self, job):
        c = job.circuit
        n_qubits = float(getattr(c, "n_qubits", 0) or c.n_qubits)
        try:
            n_2qb = float(c.n_2qb_gates())
        except Exception:
            n_2qb = 0.0
        try:
            depth = float(c.depth())
        except Exception:
            depth = 0.0
        return n_qubits, depth, n_2qb



# class QCloudBatchEnv(gym.Env):

#     metadata = {"render_modes": []}

#     def __init__(
#         self,
#         qcloud,
#         job_generator,
#         jobs_per_episode=8,
#         reward_mode="simple",
#         invalid_penalty=-5.0,
#         unscheduled_penalty=-2.0,
#         comm_cost_scale=1.0,
#     ):
#         super().__init__()
#         assert reward_mode in ["simple", "execution"]
#         self.qcloud = qcloud
#         self.job_generator = job_generator  # function that returns a list of job objects
#         self.jobs_per_episode = jobs_per_episode
#         self.reward_mode = reward_mode
#         self.invalid_penalty = invalid_penalty
#         self.unscheduled_penalty = unscheduled_penalty
#         self.comm_cost_scale = comm_cost_scale

#         # QPUs from qCloud
#         self.num_qpus = len(self.qcloud.qpus)
#         # total qubits per QPU (assume fixed for now)
#         self.total_qubits = np.array(
#             [q.ncp_qubits for q in self.qcloud.qpus], dtype=np.float32
#         )

#         # Node features: [available_qubits_norm, total_qubits_norm, degree_norm]
#         self.node_feature_dim = 3

#         # Job features: [n_qubits_norm, depth_norm, n_2qb_per_qubit_norm]
#         self.job_feature_dim = 3

#         # Observation space: Dict(node_features, job_features, action_mask)
#         self.observation_space = spaces.Dict(
#             {
#                 "node_features": spaces.Box(
#                     low=0.0,
#                     high=1.0,
#                     shape=(self.num_qpus, self.node_feature_dim),
#                     dtype=np.float32,
#                 ),
#                 "job_features": spaces.Box(
#                     low=0.0,
#                     high=1.0,
#                     shape=(self.job_feature_dim,),
#                     dtype=np.float32,
#                 ),
#                 "action_mask": spaces.Box(
#                     low=0.0,
#                     high=1.0,
#                     shape=(self.num_qpus,),
#                     dtype=np.float32,
#                 ),
#             }
#         )

#         # Multi-binary action: one bit per QPU (1 = selected)
#         self.action_space = spaces.MultiBinary(self.num_qpus)

#         # Internal episode state
#         self.jobs = None
#         self.current_job_idx = None
#         self.current_job = None
#         self.qpu_available = None  # RL’s view of available qubits
#         self.unscheduled_jobs = None
#         self.step_count = None

#         # Pre-compute degrees for node features
#         self.degrees = np.array(
#             [self.qcloud.network.degree(i) for i in range(self.num_qpus)],
#             dtype=np.float32,
#         )
#         self.max_degree = max(1.0, float(self.degrees.max()))


#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)

#         # Generate a fresh batch of jobs for this episode
#         self.jobs = self.job_generator(self.jobs_per_episode)
#         self.current_job_idx = 0
#         self.current_job = self.jobs[self.current_job_idx]

#         # Snapshot initial available qubits from qCloud
#         self.qpu_available = np.array(
#             [q.available_qubits for q in self.qcloud.qpus],
#             dtype=np.float32,
#         )

#         self.unscheduled_jobs = 0
#         self.step_count = 0

#         obs = self._build_observation()
#         info = {}
#         return obs, info

#     def step(self, action):
#         """
#         action: np.array of shape (num_qpus,), elements in {0,1}
#         """
#         self.step_count += 1

#         action = np.array(action, dtype=np.int32)
#         selected_qpus = np.where(action == 1)[0].tolist()

#         done = False
#         info = {}
#         reward = 0.0
#         invalid = False

#         job = self.current_job
#         job_n_qubits, job_depth, job_n_2qb = self._extract_job_stats(job)

#         # ----------------- Validate action -----------------

#         # 1) No QPUs selected
#         if len(selected_qpus) == 0:
#             invalid = True

#         # 2) Total available qubits among selected < job qubits
#         total_sel_qubits = float(self.qpu_available[selected_qpus].sum()) if not invalid else 0.0
#         if not invalid and total_sel_qubits < job_n_qubits:
#             invalid = True

#         if invalid:
#             reward += self.invalid_penalty
#             self.unscheduled_jobs += 1
#         else:
#             # ----------------- Apply allocation -----------------

#             # Simple greedy allocation: consume qubits from selected QPUs
#             remaining = job_n_qubits
#             # sort selected QPUs in descending order of available qubits (use larger QPUs first)
#             q_sorted = sorted(
#                 selected_qpus,
#                 key=lambda i: self.qpu_available[i],
#                 reverse=True,
#             )
#             for qi in q_sorted:
#                 if remaining <= 0:
#                     break
#                 take = min(remaining, self.qpu_available[qi])
#                 self.qpu_available[qi] -= take
#                 remaining -= take

#             # By construction it should be enough; if not, treat as invalid fallback
#             if remaining > 0:
#                 # Should not happen if checks above are right, but be safe
#                 reward += self.invalid_penalty
#                 self.unscheduled_jobs += 1
#                 invalid = True
#             else:
#                 # ----------------- Compute communication cost -----------------
#                 comm_cost = self._compute_comm_cost(job, selected_qpus)
#                 reward -= self.comm_cost_scale * comm_cost
#                 info["comm_cost"] = comm_cost

#         # ----------------- Move to next job -----------------

#         self.current_job_idx += 1
#         if self.current_job_idx >= len(self.jobs):
#             done = True
#         else:
#             self.current_job = self.jobs[self.current_job_idx]

#         # ----------------- Final penalties at end of episode -----------------
#         if done:
#             reward -= self.unscheduled_penalty * float(self.unscheduled_jobs)
#             info["unscheduled_jobs"] = self.unscheduled_jobs

#         # ----------------- Build next observation -----------------
#         obs = self._build_observation() if not done else self._build_terminal_observation()

#         return obs, reward, done, False, info

#     # ------------- Observation helpers -------------

#     def _build_observation(self):
#         """
#         Build observation dict for current job & QPU state.
#         """
#         # Node features
#         total_qubits_norm = self.total_qubits / np.maximum(self.total_qubits.max(), 1.0)
#         available_norm = self.qpu_available / np.maximum(self.total_qubits, 1.0)
#         degree_norm = self.degrees / self.max_degree

#         node_features = np.stack(
#             [available_norm, total_qubits_norm, degree_norm],
#             axis=-1,
#         ).astype(np.float32)

#         # Job features
#         job = self.current_job
#         job_n_qubits, job_depth, job_n_2qb = self._extract_job_stats(job)

#         # Normalize job features using simple heuristic constants
#         max_qubits = max(job_n_qubits, 1.0)
#         max_depth = max(job_depth, 1.0)
#         # per-qubit 2qb gate density
#         density = job_n_2qb / max(job_n_qubits, 1.0)

#         job_features = np.array(
#             [
#                 job_n_qubits / (self.total_qubits.sum() + 1e-6),  # relative to cloud size
#                 job_depth / (max_depth + 1e-6),                  # trivial normalization
#                 density / (density + 1.0),                       # squashed
#             ],
#             dtype=np.float32,
#         )

#         # Action mask: QPUs that still have qubits > 0
#         action_mask = (self.qpu_available > 0).astype(np.float32)

#         return {
#             "node_features": node_features,
#             "job_features": job_features,
#             "action_mask": action_mask,
#         }

#     def _build_terminal_observation(self):
#         # Return a "null" observation at the end of episode
#         return {
#             "node_features": np.zeros(
#                 (self.num_qpus, self.node_feature_dim), dtype=np.float32
#             ),
#             "job_features": np.zeros((self.job_feature_dim,), dtype=np.float32),
#             "action_mask": np.zeros((self.num_qpus,), dtype=np.float32),
#         }

#     def _extract_job_stats(self, job):
#         """
#         Extract basic stats from a job object.

#         Assumes:
#           job.circuit.n_qubits
#           job.circuit.n_2qb_gates()
#           job.circuit.depth()
#         """
#         circuit = job.circuit
#         n_qubits = float(circuit.n_qubits)
#         try:
#             n_2qb = float(circuit.n_2qb_gates())
#         except AttributeError:
#             n_2qb = 0.0
#         try:
#             depth = float(circuit.depth())
#         except AttributeError:
#             depth = 0.0
#         return n_qubits, depth, n_2qb

#     # ------------- Reward / cost helpers -------------

#     def _compute_comm_cost(self, job, selected_qpus):
#         """
#         Dispatch to different reward modes.
#         For now:
#           - "simple": pairwise shortest path distance avg among selected QPUs
#           - "wig": same as simple, but you will plug in your WIG-based cost here
#           - "execution": placeholder for full simulation cost
#         """
#         if len(selected_qpus) <= 1:
#             return 0.0

#         if self.reward_mode == "simple":
#             return self._simple_comm_cost(selected_qpus)

#         elif self.reward_mode == "wig":
#             # TODO: integrate real WIG-based cost.
#             # For now, use same as simple.
#             return self._simple_comm_cost(selected_qpus)

#         elif self.reward_mode == "execution":
#             # TODO: call your actual simulator here and return
#             # total execution time or some metric.
#             # Placeholder: simple comm cost proxy.
#             return self._simple_comm_cost(selected_qpus)

#         else:
#             raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

#     def _simple_comm_cost(self, selected_qpus):
#         """
#         Simple communication cost: average pairwise shortest path distance
#         between selected QPUs.
#         """
#         if len(selected_qpus) <= 1:
#             return 0.0
#         G = self.qcloud.network
#         total = 0.0
#         count = 0
#         for i in range(len(selected_qpus)):
#             for j in range(i + 1, len(selected_qpus)):
#                 u = selected_qpus[i]
#                 v = selected_qpus[j]
#                 try:
#                     d = nx.shortest_path_length(G, u, v)
#                 except nx.NetworkXNoPath:
#                     d = 1e3  # big penalty
#                 total += d
#                 count += 1
#         return total / max(count, 1)
