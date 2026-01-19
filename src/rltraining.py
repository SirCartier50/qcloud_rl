import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback

from qcloudenv import QCloudBatchEnv
from gnn_policy import QCloudFeatureExtractor
from cluster import qCloud, create_random_topology
from job import job_generator
import torch
import torch.nn as nn



class DummyJob:
    def __init__(self, job_id, circuit):
        self.id = job_id
        self.circuit = circuit


def random_circuit(num_qubits, depth):
    from pytket import Circuit
    c = Circuit(num_qubits)
    return c


def simple_job_generator(num_jobs):
    jobs = []
    for j in range(num_jobs):
        nq = np.random.randint(5, 25) 
        depth = np.random.randint(10, 100)
        c = random_circuit(nq, depth)
        jobs.append(DummyJob(j, c))
    return jobs



from stable_baselines3.common.policies import ActorCriticPolicy

class QCloudPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=QCloudFeatureExtractor,
            features_extractor_kwargs={"hidden_dim": 64},
            **kwargs,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="simple", choices=["simple", "wig", "execution"])
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--jobs_per_episode", type=int, default=8)
    parser.add_argument("--num_qpus", type=int, default=10)
    parser.add_argument("--topology_p", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    cloud = qCloud(
        num_qpus=args.num_qpus,
        topology_func=create_random_topology,
        topology_args=args.topology_p,
        ncm_qubits=5,
        ncp_qubits=30,
        need_switch=False,
    )
    jgen = job_generator()
    def make_env():
        return QCloudBatchEnv(
            qcloud=cloud,
            job_generator=jgen,
            jobs_per_episode=args.jobs_per_episode,
            reward_mode=args.mode,
        )

    env = make_vec_env(make_env, n_envs=4) 

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=256,
        n_steps=2048,
        learning_rate=3e-4,
        tensorboard_log="./qcloud_tb/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // 4,
        save_path="./checkpoints/",
        name_prefix=f"ppo_qcloud_{args.mode}",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
    )

    if args.save_path is None:
        args.save_path = f"ppo_qcloud_{args.mode}.zip"

    model.save(args.save_path)
    print(f"Saved trained model to {args.save_path}")


if __name__ == "__main__":
    main()
