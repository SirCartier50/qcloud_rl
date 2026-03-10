# qcloudenv.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from des import DES, generatingJob
from flowScheduler import flow_scheduler_1
from jobScheduler import job_scheduler
from cluster import qCloud, create_random_topology
import random
import time
from jobScheduler import placement
import math
from collections import Counter



class QCloudBatchEnv(gym.Env):
    """
    Batch RL environment for QPU selection (multi-binary actions).

    Key fixes added:
      - Episode-wise permutation of QPU indices (prevents memorizing "always pick index 5")
      - Better randomized + varied job batch generation
      - Always reset qpu availability state each episode (simple + execution)
      - Only decrement availability on the effective used QPUs
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        qcloud_creator,
        job_generator,
        jobs_per_episode=8,
        reward_mode="simple",
        invalid_penalty=-50.0,      # CHANGED default: make invalid clearly worse than typical valid costs
        unscheduled_penalty=-20.0,  # CHANGED default
        comm_cost_scale=1.0,
        unused_qpu_penalty=0.1,
        use_makespan_reward=True,
        # NEW knobs
        permute_qpus_each_episode=True,
        enforce_job_variety=True,
        max_job_sampling_tries=250,
        max_num_qpus=None,
    ):
        super().__init__()
        assert reward_mode in ["simple", "execution"]

        self.qcloud_creator = qcloud_creator
        self.qcloud = self.qcloud_creator()
        self.job_generator = job_generator
        self.jobs_per_episode = jobs_per_episode
        self.reward_mode = reward_mode
        self.invalid_penalty = float(invalid_penalty)
        self.unscheduled_penalty = float(unscheduled_penalty)
        self.comm_cost_scale = float(comm_cost_scale)
        self.unused_qpu_penalty = float(unused_qpu_penalty)
        self.use_makespan_reward = bool(use_makespan_reward)

        self.permute_qpus_each_episode = bool(permute_qpus_each_episode)
        self.enforce_job_variety = bool(enforce_job_variety)
        self.max_job_sampling_tries = int(max_job_sampling_tries)

        self.current_num_qpus = len(self.qcloud.qpus)
        if max_num_qpus is None:
            self.max_num_qpus = self.current_num_qpus
        else:
            self.max_num_qpus = int(max_num_qpus)
            if self.current_num_qpus > self.max_num_qpus:
                raise ValueError(f"Initial cloud size {self.current_num_qpus} > max_num_qpus {self.max_num_qpus}")

        self.scheduler = job_scheduler(None, None, self.qcloud, scheduler_type="default")

        self.num_qpus = len(self.qcloud.qpus)
        # Maps observation space index to real QPU index. -1 means padded/empty.
        self.obs_to_real_map = np.full(self.max_num_qpus, -1, dtype=np.int32)

        # permutation maps (obs/action index -> real qpu index)
        self.perm = np.arange(self.num_qpus, dtype=np.int32)
        self.inv_perm = np.arange(self.num_qpus, dtype=np.int32)

        # capacities (recomputed on reset in case you later make heterogeneous clouds)
        self.total_qubits = np.array([q.ncp_qubits for q in self.qcloud.qpus], dtype=np.float32)

        # features
        if self.reward_mode == "execution":
            self.node_feature_dim = 5
        else:
            self.node_feature_dim = 3
        self.job_feature_dim = 3

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_qpus, self.node_feature_dim),
                    shape=(self.max_num_qpus, self.node_feature_dim),
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
                    shape=(self.max_num_qpus,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.MultiBinary(self.num_qpus)
        self.action_space = spaces.MultiBinary(self.max_num_qpus)

        # episode state
        self.jobs = None
        self.current_job_idx = None
        self.current_job = None
        self.qpu_available = None
        self.unscheduled_jobs = None
        self.step_count = None

        self.degrees = np.array([self.qcloud.network.degree(i) for i in range(self.num_qpus)], dtype=np.float32)
        self.degrees = np.array([self.qcloud.network.degree(i) for i in range(self.current_num_qpus)], dtype=np.float32)
        self.max_degree = max(1.0, float(self.degrees.max()))

    def _refresh_capacity_state(self):
        """
        Reset per-episode availability state (numpy only).
        We DO NOT need to mutate real QPU objects for simple reward,
        but we do want consistent feasibility/masks across the episode.
        """
        # refresh capacities (in case cloud becomes heterogeneous later)
        self.total_qubits = np.array([q.ncp_qubits for q in self.qcloud.qpus], dtype=np.float32)
        self.qpu_available = self.total_qubits.copy().astype(np.float32)

    def _sample_varied_jobs(self, seed=None):
        """
        Force a mixture of job sizes relative to max single-QPU capacity.
        Helps the agent learn both "single QPU" and "must partition" cases.
        """
        # baseline capacity for categorization
        cap = int(max(self.total_qubits.max(), 1.0))

        def job_nq(job):
            c = job.circuit
            return int(getattr(c, "n_qubits", 0) or c.n_qubits)

        jobs = []
        tries = 0

        # target mix (rough)
        # - some <= cap (fits one QPU)
        # - many between (cap, 3*cap] (needs 2-3 QPUs)
        # - avoid > 3*cap (often impossible and teaches "be invalid")
        while len(jobs) < self.jobs_per_episode and tries < self.max_job_sampling_tries:
            tries += 1
            j = self.job_generator.generate_random_job(num_jobs=1)[0]
            nq = job_nq(j)

            if nq <= cap:
                accept = (random.random() < 0.35)
            elif nq <= 2 * cap:
                accept = (random.random() < 0.55)
            elif nq <= 3 * cap:
                accept = (random.random() < 0.85)
            else:
                accept = False

            if accept:
                jobs.append(j)

        if len(jobs) < self.jobs_per_episode:
            # fallback: just generate a full batch
            jobs = self.job_generator.generate_random_job(num_jobs=self.jobs_per_episode)

        random.shuffle(jobs)
        return jobs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # make randomness actually change across episodes (also fixes same-job repeats)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.qcloud = self.qcloud_creator()
        if len(self.qcloud.qpus) != self.num_qpus:
            raise ValueError(f"qcloud_creator returned {len(self.qcloud.qpus)} QPUs, expected {self.num_qpus}")
        self.current_num_qpus = len(self.qcloud.qpus)
        if self.current_num_qpus > self.max_num_qpus:
            raise ValueError(f"qcloud_creator returned {self.current_num_qpus} QPUs, but max is {self.max_num_qpus}")

        self.degrees = np.array([self.qcloud.network.degree(i) for i in range(self.num_qpus)], dtype=np.float32)
        self.degrees = np.array([self.qcloud.network.degree(i) for i in range(self.current_num_qpus)], dtype=np.float32)
        self.max_degree = max(1.0, float(self.degrees.max()))

        if self.scheduler:
            self.scheduler.qcloud = self.qcloud
        
        # refresh availability & capacities every episode (simple + execution)
        self._refresh_capacity_state()

        # permute QPU indices each episode to prevent index memorization
        # SCATTER STRATEGY: map real QPUs to random slots in the observation space
        self.obs_to_real_map.fill(-1)
        if self.permute_qpus_each_episode:
            self.perm = np.random.permutation(self.num_qpus).astype(np.int32)
            self.inv_perm = np.empty_like(self.perm)
            self.inv_perm[self.perm] = np.arange(self.num_qpus, dtype=np.int32)
            active_slots = np.random.choice(self.max_num_qpus, self.current_num_qpus, replace=False)
        else:
            self.perm = np.arange(self.num_qpus, dtype=np.int32)
            self.inv_perm = np.arange(self.num_qpus, dtype=np.int32)
            active_slots = np.arange(self.current_num_qpus)
        self.obs_to_real_map[active_slots] = np.arange(self.current_num_qpus)

        # jobs: random + varied distribution (optional but recommended)
        if self.enforce_job_variety:
            self.jobs = self._sample_varied_jobs(seed=seed)
        else:
            self.jobs = self.job_generator.generate_random_job(num_jobs=self.jobs_per_episode)
            random.shuffle(self.jobs)

        self.current_job_idx = 0
        self.current_job = self.jobs[0]
        self.unscheduled_jobs = 0
        self.step_count = 0

        obs = self._build_observation()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.array(action, dtype=np.int32)

        # action is in OBS index space; map to REAL qpu indices
        selected_obs_idx = np.where(action == 1)[0].astype(np.int32)
        selected_qpus = self.perm[selected_obs_idx].tolist()
        selected_obs_indices = np.where(action == 1)[0]

        # Map selected observation indices to real QPU indices, filtering out padded ones
        selected_qpus = []
        for obs_idx in selected_obs_indices:
            real_idx = self.obs_to_real_map[obs_idx]
            if real_idx != -1:
                selected_qpus.append(real_idx)

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
        else:
            total_sel_qubits = float(self.qpu_available[selected_qpus].sum())
            if total_sel_qubits < job_nq:
                invalid = True

        if invalid:
            reward += self.invalid_penalty
            self.unscheduled_jobs += 1
        else:
            remaining = float(job_nq)
            q_sorted = sorted(
                selected_qpus,
                key=lambda idx: float(self.qpu_available[idx]),
                reverse=True,
            )

            # shrink selection to only the QPUs actually needed to meet demand
            used_qpus = []
            cap_sum = 0.0
            for qi in q_sorted:
                if cap_sum >= remaining:
                    break
                avail = float(self.qpu_available[qi])
                if avail > 0:
                    used_qpus.append(qi)
                    cap_sum += avail

            if cap_sum < remaining or len(used_qpus) == 0:
                reward += self.invalid_penalty
                self.unscheduled_jobs += 1
            else:
                # discourage selecting extra unused qpus
                extra = max(0, len(selected_qpus) - len(used_qpus))
                reward -= self.unused_qpu_penalty * float(extra)

                # decrement only the QPUs we actually use
                remaining = float(job_nq)
                for qi in used_qpus:
                    if remaining <= 0:
                        break
                    can_take = min(remaining, float(self.qpu_available[qi]))
                    self.qpu_available[qi] -= can_take
                    remaining -= can_take

                if remaining > 0:
                    reward += self.invalid_penalty
                    self.unscheduled_jobs += 1
                else:
                    circuit = job.circuit
                    wig = self.scheduler.convert_to_weighted_graph(circuit)

                    k_eff = len(used_qpus)
                    wig_nodes = int(wig.number_of_nodes())

                    # Prevent partitioning failure modes
                    if k_eff > max(1, wig_nodes):
                        reward += self.invalid_penalty
                        self.unscheduled_jobs += 1
                        invalid = True
                    else:
                        if self.reward_mode == "simple":
                            # Build placement using k_eff (effective QPUs)
                            if k_eff == 1:
                                qpu_list = [self.qcloud.qpus[used_qpus[0]]]      # LIST OF QPU OBJECTS
                                place = placement(job, 1, qpu_list, wig)

                                # match heuristic behavior
                                place.communication_cost = 0

                                # IMPORTANT for execution mode:
                                # heuristic sets time/modified_circuit; do the same
                                place.modified_circuit = circuit
                                try:
                                    # if placement has a method for single case
                                    place.get_time(1)
                                except Exception:
                                    # leave time as-is if simulator fills it later
                                    pass
                            else:
                                parts = self.scheduler.partition_circuit(k_eff, wig)
                                labels = sorted(set(parts[0]))
                                place = placement(job, parts, [{j: used_qpus[i] for i, j in enumerate(labels)}], wig)
                                place.get_communication_cost(wig)
                                place.get_time(parts)
                            if getattr(place, "partition", 1) == 1:
                                # single-qpu => comm cost 0
                                reward -= 0.0
                            else:
                                # function mutates place.communication_cost
                                if hasattr(place, "communication_cost"):
                                    place.communication_cost = 0.0
                                place.get_communication_cost(wig)
                                reward -= self.comm_cost_scale * float(getattr(place, "communication_cost", 0.0))
                        else:
                            # Build placement using k_eff (effective QPUs)
                            if k_eff == 1:
                                qpu_list = [self.qcloud.qpus[used_qpus[0]]]      # LIST OF QPU OBJECTS
                                place = placement(job, 1, qpu_list, wig)

                                # match heuristic behavior
                                place.communication_cost = 0

                                # IMPORTANT for execution mode:
                                # heuristic sets time/modified_circuit; do the same
                                place.modified_circuit = circuit
                                try:
                                    # if placement has a method for single case
                                    place.get_time(1)
                                except Exception:
                                    # leave time as-is if simulator fills it later
                                    pass
                            else:
                                # parts = self.scheduler.partition_circuit(k_eff, wig)
                                # labels = sorted(set(parts[0]))
                                # place = placement(job, parts, [{j: used_qpus[i] for i, j in enumerate(labels)}], wig)
                                # place.get_communication_cost(wig)
                                # place.get_time(parts)
                                # make sure we don't exceed available qpus
                                parts = self.scheduler.partition_circuit(k_eff, wig)
                                print("St:",parts)
                                comb = {k_eff: parts}
                                count = Counter(parts[0])
                                labels = list(count.keys())

                                # must match how many partitions exist
                                if len(labels) != len(used_qpus):
                                    invalid = True
                                    assert False
                                else:
                                    mapping = {}
                                    for i, label in enumerate(labels):
                                        mapping[label] = used_qpus[i]

                                    place = placement(job, comb[k_eff], [mapping], wig)
                                    place.get_communication_cost(wig)
                                    place.get_time(parts)

                            # execution-mode path: use estimated time as cost
                            if getattr(place, "time", None) is None:
                                place.time = 0.0
                            
                            if place.time == 0.0 and job_depth > 0:
                                place.time = job_depth
                            if invalid:
                                reward += self.invalid_penalty
                                self.unscheduled_jobs += 1
                            else:
                                # execution-mode path: use estimated time as cost
                                if getattr(place, "time", None) is None:
                                    place.time = 0.0
                                
                                if place.time == 0.0 and job_depth > 0:
                                    place.time = job_depth
                            
                                reward -= float(place.time)



                        

        # advance job
        self.current_job_idx += 1
        if self.current_job_idx >= len(self.jobs):
            done = True
        else:
            self.current_job = self.jobs[self.current_job_idx]

        if done:
            # if self.reward_mode == "execution":
            #     self.flow_scheduler.run()
            #     print("finished flow")
            #     self.des.run()
            #     if self.use_makespan_reward:
            #         end_times = self.des_logger.get_end_times()
            #         makespan = max(end_times) if end_times else 0.0
            #         reward -= float(makespan)
            #         info["makespan"] = float(makespan)
            #     else:
            #         jcts = self.des_logger.get_jcts()
            #         sum_jct = float(sum(jcts)) if jcts else 0.0
            #         reward -= sum_jct
            #         info["sum_jct"] = float(sum_jct)

            reward -= self.unscheduled_penalty * float(self.unscheduled_jobs)
            info["unscheduled_jobs"] = int(self.unscheduled_jobs)

        obs = self._build_observation() if not done else self._build_terminal_observation()
        return obs, float(reward), done, truncated, info

    def _build_observation(self):
        # normalize
        total_qubits_norm = self.total_qubits / np.maximum(self.total_qubits.max(), 1.0)
        available_norm = self.qpu_available / np.maximum(self.total_qubits, 1.0)
        degree_norm = self.degrees / self.max_degree

        if self.reward_mode == "execution":
            occupied = np.array([1.0 if q.occupied else 0.0 for q in self.qcloud.qpus], dtype=np.float32)

            running_counts = []
            for q in self.qcloud.qpus:
                cnt = 0
                for _, st in getattr(q, "job_status", {}).items():
                    if st == "running":
                        cnt += 1
                running_counts.append(cnt)
            running_counts = np.array(running_counts, dtype=np.float32)
            running_norm = running_counts / max(1.0, float(self.jobs_per_episode))

            node_features_real = np.stack(
                [available_norm, total_qubits_norm, degree_norm, occupied, running_norm],
                axis=-1
            ).astype(np.float32)
        else:
            node_features_real = np.stack(
                [available_norm, total_qubits_norm, degree_norm],
                axis=-1
            ).astype(np.float32)

        job = self.current_job
        n_qubits, depth, n_2qb = self._extract_job_stats(job)

        rel_qubits = n_qubits / (float(self.total_qubits.sum()) + 1e-6)
        rel_depth = depth / (depth + 1.0)
        density = n_2qb / max(n_qubits, 1.0)
        density = density / (density + 1.0)

        job_features = np.array([rel_qubits, rel_depth, density], dtype=np.float32)

        # action mask is also in real index space initially
        action_mask_real = (self.qpu_available > 0).astype(np.float32)

        # present in permuted obs index space
        node_features = node_features_real[self.perm]
        action_mask = action_mask_real[self.perm]
        # PADDING & SCATTERING: Create fixed-size buffers and fill them
        node_features_padded = np.zeros((self.max_num_qpus, self.node_feature_dim), dtype=np.float32)
        action_mask_padded = np.zeros((self.max_num_qpus,), dtype=np.float32)

        # Find which observation slots are active and which real QPUs they map to
        valid_obs_slots = np.where(self.obs_to_real_map != -1)[0]
        real_indices_for_slots = self.obs_to_real_map[valid_obs_slots]

        node_features_padded[valid_obs_slots] = node_features_real[real_indices_for_slots]
        action_mask_padded[valid_obs_slots] = action_mask_real[real_indices_for_slots]

        return {
            "node_features": node_features,
            "node_features": node_features_padded,
            "job_features": job_features,
            "action_mask": action_mask,
            "action_mask": action_mask_padded,
        }

    def _build_terminal_observation(self):
        return {
            "node_features": np.zeros((self.num_qpus, self.node_feature_dim), dtype=np.float32),
            "node_features": np.zeros((self.max_num_qpus, self.node_feature_dim), dtype=np.float32),
            "job_features": np.zeros((self.job_feature_dim,), dtype=np.float32),
            "action_mask": np.zeros((self.num_qpus,), dtype=np.float32),
            "action_mask": np.zeros((self.max_num_qpus,), dtype=np.float32),
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
