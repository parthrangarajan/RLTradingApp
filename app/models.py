import numpy as np
from stable_baselines import A2C, DQN, PPO2


def make_model(algo, env, cfg):
    """
    Build one of the three candidate learning algorithms with settings suited to this problem.

    algo: "a2c", "ppo", or "dqn"
    env:  a stable-baselines environment (wrapped in DummyVecEnv)
    cfg:  the run settings, which supply the learning rate and the seed
    """
    lr = cfg["learning_rate"]
    seed = cfg["seed"]

    if algo == "a2c":
        return A2C(
            "MlpPolicy", env,
            learning_rate=lr,
            ent_coef=0.01,        # entropy bonus: keeps the agent exploring
            seed=seed,
            verbose=0,
        )

    if algo == "ppo":
        return PPO2(
            "MlpPolicy", env,
            learning_rate=lr,
            n_steps=256,          # days collected before each update
            nminibatches=4,       # 256 / 4 = 64 days per mini-update
            ent_coef=0.01,
            cliprange=0.2,        # the epsilon clip from the concept notes
            seed=seed,
            verbose=0,
        )

    if algo == "dqn":
        return DQN(
            "MlpPolicy", env,
            learning_rate=lr,
            buffer_size=50000,            # how many past days the replay buffer holds
            learning_starts=500,          # collect this many days before learning begins
            batch_size=64,
            gamma=0.99,                   # how much future reward counts (between 0 and 1)
            exploration_fraction=0.3,     # share of training spent trying random actions
            exploration_final_eps=0.05,   # final chance of a random action
            seed=seed,
            verbose=0,
        )

    raise ValueError("Unknown algorithm: " + algo)


def act(model, obs):
    """Ask any supported model for its action, always choosing the most likely one."""
    action, _ = model.predict(obs, deterministic=True)
    return int(np.asarray(action).reshape(-1)[0])