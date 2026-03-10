import argparse
import numpy as np
import torch
import os
import json

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from qcloudenv import QCloudBatchEnv
from gnn_policy import QCloudFeatureExtractor
from cluster import qCloud, create_random_topology
from job import job_generator


def resolve_use_gnn(use_gnn_arg: str, mode: str) -> bool:
    if use_gnn_arg == "auto":
        return mode == "execution"
    if use_gnn_arg == "1":
        return True
    if use_gnn_arg == "0":
        return False
    raise ValueError("--use_gnn must be one of: auto, 0, 1")


def build_edge_index_from_nx(G):
    edges = list(G.edges())
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def mask_fn(env: QCloudBatchEnv):
    """
    For MultiBinary(n), MaskablePPO expects mask shape (n, 2):
      mask[i,0] => allow 0
      mask[i,1] => allow 1
    Always allow 0, allow 1 if action_mask[i] is True.
    """
    obs = env._build_observation()
    m = obs["action_mask"].astype(bool)
    return np.stack([np.ones_like(m, dtype=bool), m], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="simple", choices=["simple", "execution"])
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs_per_episode", type=int, default=8)
    parser.add_argument("--num_qpus", type=int, default=10)
    parser.add_argument("--max_num_qpus", type=int, default=None, help="Size of the action space (deployment size)")
    parser.add_argument("--max_num_qpus", type=int, default=None, help="Max size of action/observation space (for variable-size clouds)")
    parser.add_argument("--topology_p", type=float, default=0.5)
    parser.add_argument("--use_gnn", type=str, default="auto", choices=["auto", "0", "1"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    use_gnn = resolve_use_gnn(args.use_gnn, args.mode)
    
    # If max_num_qpus is not set, default to num_qpus (fixed size training)
    if args.max_num_qpus is None:
        args.max_num_qpus = args.num_qpus

    if args.run_name is None:
        args.run_name = (
            f"maskppo_{args.mode}"
            f"_gnn{int(use_gnn)}"
            f"_q{args.num_qpus}"
            f"_jobs{args.jobs_per_episode}"
            f"_lr{args.learning_rate:g}"
            f"_nsteps{args.n_steps}"
            f"_bs{args.batch_size}"
            f"_seed{args.seed}"
        )

    run_dir = os.path.join(args.models_dir, args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cfg = vars(args).copy()
    cfg["use_gnn_resolved"] = use_gnn
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # template cloud ONLY for edge_index
    template_cloud = qCloud(
        num_qpus=args.num_qpus,
        num_qpus=args.max_num_qpus,
        topology_func=create_random_topology,
        topology_args=args.topology_p,
        ncm_qubits=5,
        ncp_qubits=30,
        need_switch=False,
    )
    edge_index = build_edge_index_from_nx(template_cloud.network)

    # Build a list of 0-arg env constructors (one per env)
    env_fns = []
    for rank in range(args.n_envs):
        def make_one_env(rank=rank):
            def qcloud_creator():
                return qCloud(
                    num_qpus=args.num_qpus,
                    num_qpus=np.random.randint(4, args.num_qpus + 1), # Generate clouds up to args.num_qpus
                    num_qpus=np.random.randint(4, args.num_qpus + 1),
                    topology_func=create_random_topology,
                    topology_args=args.topology_p,
                    ncm_qubits=np.random.randint(2, 10),
                    ncp_qubits=np.random.randint(10, 40),
                    need_switch=False,
                )
            jgen = job_generator()

            env = QCloudBatchEnv(
                qcloud_creator=qcloud_creator,
                job_generator=jgen,
                jobs_per_episode=args.jobs_per_episode,
                reward_mode=args.mode,
                max_num_qpus=args.max_num_qpus,
            )

            env = ActionMasker(env, mask_fn)

            # per-env deterministic seed
            env.reset(seed=args.seed + 1000 * rank)
            return env

        env_fns.append(make_one_env)

    env = DummyVecEnv(env_fns)

    policy_kwargs = None
    if use_gnn:
        policy_kwargs = dict(
            features_extractor_class=QCloudFeatureExtractor,
            features_extractor_kwargs={
                "hidden_dim": args.hidden_dim,
                "edge_index": edge_index,
                "job_condition_nodes": True,
            },
        )

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50_000 // max(1, args.n_envs)),
        save_path=ckpt_dir,
        name_prefix="ckpt",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()

