# qcloud_rl

when making a virtual environment make sure that it uses python version 3.9 since this is, up to my knowledge, the only version that allows you to install all the required packages. (some versions don't all some packages to install)

when virtual env is ready, run the install.py file to install all needed packages.

# rltraining.py:
  The main training file to generate a model. Contains default parameters but accepts from the command line.
  * --mode, type=str, default="simple", choices=["simple", "execution"]
  * --total_timesteps, type=int, default=500_000
  * --jobs_per_episode, type=int, default=8
  * --num_qpus, type=int, default=10
  * --topology_p, type=float, default=0.5
  * --save_path, type=str, default=None
# qcloudenv.py:
  The environment used for the model to train in.
# other files
  Most of the other files are source code from the qcloud simulator repo that I've altered a bit. These are neccesary for the main code since it pulls several functions from it.
  
