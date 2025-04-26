from env.flag_frenzy_env import FlagFrenzyEnv
from env.SimulationInterface import ControllableEntityManouver

def sim_test():
    env = FlagFrenzyEnv()

    try:
        while env.compute_reward() == 0:
            env._tick()
            env._get_observations()

            # Example on how to perform an attack
            flagship = env.find_entity_by_name("Renhai")
            b1 = env.find_entity_by_name("B1")

            if (flagship.IsAlive()):
                if b1.CurrentManouver != ControllableEntityManouver.Combat:
                    env.execute_action([3, [b1.EntityId / env.max_entities, flagship.EntityId / env.max_entities, 0.0, 0.0]])
            elif b1.CurrentManouver == ControllableEntityManouver.NoManouver:
                env.execute_action([2, [b1.EntityId / env.max_entities]])
        print("Successfully ran simulation.")

    except Exception as e:
        print(f"There was error running the game simulation! {e}")

def gym_env_test():
    env = FlagFrenzyEnv()

    try:
        obs, info = env.reset()

        for i in range(5):
            action = env.action_space.sample()
            observation, reward, term, trunc, info = env.step(action)

        env.close()
        print("Successfully stepped through gymansium environment!")

    except Exception as e:
        print(f"There was an error running the game simulation! {e}")

if __name__ == "__main__":
    print("Running simulation test...")
    sim_test()
    print("Simulation test completed.")

    print("Running gymnasium test...")
    gym_env_test()
    print("Gymansium test completed.")