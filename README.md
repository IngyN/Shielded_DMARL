# Shielded_DMARL
This repository contains the code for the composed shielding experiments with maddpg [Safe multi-agent reinforcement learning via shielding](https://arxiv.org/pdf/2101.11196) paper.

#### Prerequisites:
- Python 3.6+
- gym
- matplotlib 3.0.0 
- [particle environment](https://github.com/openai/multiagent-particle-envs) for deep MARL experiments (modified to be discretized + scenarios - code missing due to computer problems).

#### Code structure:
- `GridShield.py`: contains the implementation of the composed shielding method currently restricted to 2 agents per shield but code can be modified to accomodate more. 
- `train_maddpg.py`: train for a given scenario and record information with shielding using composed shielding option. 
- `train_test.py`: train then run testing phase (no learning or exploration) and record relevant information. 
- `/logs`: contains the output logged
- `/policy`: contains policy checkpoints for maddpg
- `/learning_curves`: contains relevant info for graphing (rewards and collisions).
- `/benchmark_files`: contains info pertaining to collisions for shielding and without shielding. 

#### Notes:
- Code is provided as is and not actively maintained at the moment. However, I am happy to answer questions.

