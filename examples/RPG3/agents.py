class Agent:
    kind = "agent"  # class variable shared by all instances

    def __init__(self, model):
        self.health = 10  # for the agents, this is how hungry they are
        self.appearance = [0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0]  # agents are blue
        self.vision = 4  # agents can see three radius around them
        self.model = model  # agent model here. need to add a tad that tells the learning somewhere that it is DQN
        self.value = 0  # agents have no value
        self.reward = 0  # how much reward this agent has collected
        self.location = None
        self.passable = 0  # whether the object blocks movement
        self.episode_memory = deque([], maxlen=100)  # we should read in these maxlens
        self.has_transitions = True
        self.action_type = "neural_network"

    def init_replay(self, numberMemories, pov_size=9, visual_depth=3):
        """
        Fills in blank images for the LSTM before game play.
        Implicitly defines the number of sequences that the LSTM will be trained on.
        """
        # pov_size = 9 # this should not be needed if in the input above
        image = torch.zeros(1, numberMemories, 7, pov_size, pov_size).float()
        priority = torch.tensor(0.1)
        blank = torch.tensor(0.0)
        exp = (priority, (image, blank, blank, image, blank))
        self.episode_memory.append(exp)

    def movement(self, action):
        """
        Takes an action and returns a new location
        """
        if action == 0:
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == 1:
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == 2:
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == 3:
            new_location = (self.location[0], self.location[1] + 1, self.location[2])
        return new_location

    def transition(self, env, models, action, location):
        """
        Changes the world based on the action taken
        """
        done = 0
        reward = 0
        attempted_location = self.movement(action)
        holdObject = env.world[location]

        if env.world[attempted_location].passable == 1:
            env.world[location] = EmptyObject()
            reward = env.world[attempted_location].value
            env.world[attempted_location] = holdObject
            new_loc = attempted_location

        else:
            if isinstance(
                env.world[attempted_location], Wall
            ):  # Replacing comparison with string 'kind'
                reward = -0.1

        next_state = env.pov(new_loc)
        self.reward += reward

        return env.world, reward, next_state, done, new_loc