from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


single_agent_reward = [1.]
single_agent_rewards = [0, 1., -1., 50., 0]
multi_agent_reward = [0., -1.,  1.,  0., -1.,  0.,  -50., -1., 0, 0]
dones = [1 for _ in range(10)]

agent_dict = {"indivudal_rewards": [], "beam_fired": [], "beam_hit": [], "apples_consumed": []}


class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.individual_rewards: List[float] = []
        self.beam_fired: List[int] = []
        self.beam_hit: List[int] = []
        self.apples_consumed: List[int] = []

    # The below two methods are for item assignment (calling and setting things like with a dict -> dict["key"] = value)
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


def create_agents(num_agents) -> Dict[str, Agent]:
    agents = {}
    for i in range(num_agents):
        agent_id = f"agent-{str(i)}"
        agents[agent_id] = Agent(agent_id)

    # print(agents)
    # print(agents["agent-0"])
    # agents["agent-0"].beam_fired = [1]
    # print(agents)
    # print(agents["agent-0"]['beam_fired'])

    return agents


def psuedo_step(rewards, dones, step_size=5):
    agents = create_agents(5)
    print("Here are the rewards:", rewards)

    for i in range(step_size):
        for j in range(i, len(dones), step_size):
            print(f"i: {i} - rewards[i]: {rewards[j]} - j: {j} - dones[j]: {dones[j]}")
            agent_id = f"agent-{str(i)}" # Probably change this for the real thing - make it a pure int as a key instead of needing to recreate it as a string

            if dones[j]:
                # Individual agent metrics.
                # Note that this wouldn't include metrics from the last step!
                # agent_id = j % 5
                agents[f"{agent_id}"]["individual_rewards"] += [rewards[j]]
                if rewards[j] == -1:
                    agents[f"{str(agent_id)}"]["beam_fired"] += [1]
                elif rewards[j] == 1:
                    agents[f"{str(agent_id)}"]["apples_consumed"] += [1]
                elif rewards[j] == -50:
                    agents[f"{str(agent_id)}"]["beam_hit"] += [1]

    # print the agents dict
    for agent_id, agent_object in agents.items():
        print(f"\nAgent {agent_object.__dict__}")

    print(agents)


def testing_ts2xy():
    log_dir = "./logs/vec_monitor_logs"
    results = load_results(log_dir)
    print(results)
    x, y = ts2xy(results, 'episodes')
    print("x", x)
    print("y", y)
    timesteps = 1e5
    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
    # plt.show()
    print("my own stuff")
    print(results.x.values[-18:]) # Now all I need to do is use the # agents x # envs to get the right number of rows
    print("type of results.x.values", type(results.x.values))


if __name__ == '__main__':
    # psuedo_step(multi_agent_reward, dones, step_size=5)
    testing_ts2xy()

