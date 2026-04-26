# CEIA Baseline Agent

**Agent name:** Team 52 agent

**Author (s):** Minwoo Cho (mcho318@gatech.edu)

## Description
An agent is trained using PPO with reward shaping while competing against strong baselines and through self-play. I propose a novel training strategy that initially uses a pre-trained CEIA baseline as the opponent policy, then gradually transitions to self-play so that the agent can improve by competing against its own previous versions. Reward shaping is incorporated throughout this process to facilitate more efficient and stable training.

"train_ray_minwoo.py" implements the training architecture that introduces the above training concepts.
"util.py" implements the reward shaping components.


