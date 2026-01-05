from src.environment.gym_wrapper import SimulatedWorldEnv

env = SimulatedWorldEnv(grid_size=6, seed=1)
obs, _ = env.reset()

for _ in range(25):
    obs, reward, term, trunc, _ = env.step(5)  # force interaction
    env.render()
    print(f"Reward: {reward:.3f}")
