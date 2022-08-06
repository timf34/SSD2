from typing import List

single_agent_reward = [1.]
single_agent_rewards = [0, 1., -1., 50., 0]
multi_agent_reward = [0., -1.,  1.,  0., -1.,  0.,  -50., -1., 0, 0]
dones = [1 for i in range(10)]

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


def create_agents(num_agents) -> dict:
    agents = {}
    for i in range(num_agents):
        agent_id = f"agent-{str(i)}"
        agents[agent_id] = Agent(agent_id)

    print(agents)
    print(agents["agent-0"])
    agents["agent-0"].beam_fired = [1]
    print(agents)
    print(agents["agent-0"]['beam_fired'])

    return agents


def psuedo_step(rewards, dones, step_size):
    agents = create_agents(5)


if __name__ == '__main__':
    psuedo_step(multi_agent_reward, dones, step_size=5)
