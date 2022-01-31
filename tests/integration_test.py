from core.agent import RLAgent
from core.environment import KagomeLatticeEnv

if __name__ == '__main__':
    env = KagomeLatticeEnv(n_sq_cells=2)
    agent = RLAgent(env, n_hidden=10, learning_rate=0.0005)

    lattice = env.reset()
    policy = agent.policy_network

    print(agent.policy_network(lattice))

    # losses = agent.learn(run_n_trajectories=2, train_n_epochs=2)
