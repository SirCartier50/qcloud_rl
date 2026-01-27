# qcloud_rl

This repo contains source code for research on applying reinforcement learning to the initial QPU mapping stage of scheduling quantum jobs.

The idea is to train a model (currently Proximal Policy Optimization, PPO) to learn and provide the best placement of QPUs for a job given the current state of the qcloud. By creating an environment that simulates the job scheduling process of qcloud, the model selects QPUs that it thinks would be able to do the job and learns to improve its selections based on either the communication cost of the selection or the execution time of the job after selecting.

Over time, the model aims to maximize reward, which corresponds to minimizing communication cost or execution time. There are also penalties for invalid selections of QPUs (e.g., when the selected QPUs do not have enough resources to run the job).

---

## Setup

### Python Version

When making a virtual environment, make sure that it uses **Python 3.9**, since this is, to my knowledge, the only version that allows all required packages to be installed. Some Python versions do not allow certain packages to install correctly.

### Installing Dependencies

Once the virtual environment is ready, run:

```bash
python install.py
```

## rltraining.py:
  The main training file to generate a model. Contains default parameters but accepts from the command line.
  * --mode, type=str, default="simple", choices=["simple", "execution"]
  * --total_timesteps, type=int, default=500_000
  * --jobs_per_episode, type=int, default=8
  * --num_qpus, type=int, default=10
  * --topology_p, type=float, default=0.5
  * --save_path, type=str, default=None
## qcloudenv.py:
  The environment used for the model to train in.
## other files
  Most of the other files are source code from the qcloud simulator repo that I've altered a bit. These are neccesary for the main code since it pulls several functions from it.

* This repo contains source code for research on applying reinforcement learning to the intial qpu mapping stage of scheduling quantum jobs. The idea is to train a model (currently Proximal Polcy Optimization or PPO) to learn and try to provide the best placement of qpus for a job given the current state of the qcloud. By creating an environment that simulates the job scheduling process of qcloud, the model selects qpus that it thinks would be able to do the job and can learn to better select them based on either the communication cost of the selection or the execution time of the job after selecting. Over time the model wants to get the best reward possible which would be the result of either of the two. There's also penalties for invalid selection of qpus in which they don't have enough resources for the job.

## current issues
  * When having more than one selected qpu, the job gets partitioned but the issue arises due to one of the functions that is called to do the paritioning. Before partitioning, a weighted graph (wig) representation of the job is generated to then be partitioned n amount of times (n being the number of selected qpus) but would sometimes provide a warning in which it can't partition a wig of 0 vertices. I've also noticed that sometimes the number of nodes in the wig is way less than the amount of selected qpus.
  
