import numpy as np
from numpy.random import rand

from social_dilemmas.envs.agent import BoxTrappedAgent
from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.maps import BOX_TRAPPED_MAP

# For consistency with the other environments, I'll also use "FIRE" for now, but we can probs remove it in this env tbh
_BOX_TRAPPED_ACTIONS = {"FIRE": 5, "UNLOCK": 1}  # length of unlocking beam (we won't actually see it anyways)
BOX_TRAPPED_VIEW_SIZE = 7

APPLE_RADIUS = 2


class BoxTrapped(MapEnv):
    def __init__(
            self,
            ascii_map=BOX_TRAPPED_MAP,
            num_agents=2,
            return_agent_actions=True,
            use_collective_reward=False,
            inequity_averse_reward=False,
            alpha=0.0,
            beta=0.0,
    ):
        super().__init__(
            ascii_map=ascii_map,
            extra_actions=_BOX_TRAPPED_ACTIONS,
            view_len=BOX_TRAPPED_VIEW_SIZE,
            num_agents=num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            beta=beta,
            alpha=alpha,
        )
        self.apple_points = []
        self.box_is_locked: bool = True
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])

    @property
    def action_space(self):
        return DiscreteWithDType(9, dtype=np.int8)

    def custom_action(self, agent, action):
        updates = []
        if action == "FIRE":
            agent.fire_beam(b"F")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"F",
            )
        elif action == "UNLOCK":
            pass
        return updates

    def custom_reset(self):
        """Initialize the walls, the apples and lock the box."""

        self.box_is_locked = True

        # TODO: ensure that the agent spawns in the correct place.
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")


    def setup_agents(self):
        # Initializing the agents.
        map_with_agents = self.get_map_with_agents()

        for i in range(self._num_agents):
            agent_id = f"agent-{str(i)}"
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = BoxTrappedAgent(agent_id, spawn_point, rotation, grid, view_len=BOX_TRAPPED_VIEW_SIZE)
            self.agents[agent_id] = agent

    def custom_map_update(self):
        # Spawns the apples every step if there are any.
        new_apples = self.spawn_apples()  # CHECK THE REGROW RATE ON THIS! I want it to be basically instant!
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step. Note that apples spawn 100% of the time (don't care about apple
        density).

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        agent_positions = self.agent_pos
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                new_apple_points.append((row, col, b"A"))
        return new_apple_points
