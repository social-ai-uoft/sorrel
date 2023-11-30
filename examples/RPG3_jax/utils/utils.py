import numpy as np
from gem.models.perception import agent_visualfield
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt


def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec


def find_moveables(world):
    # see if the last places this is being used this is needed
    moveList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                if world[i, j, k].static == 0:
                    moveList.append((i, j, k))
    return moveList


def find_agents(world):
    # update gems and wolves to use find_instance instead of this and remove this code
    agentList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                if world[i, j, k].kind == "agent":
                    agentList.append((i, j, k))
    return agentList


def find_instance(world, kind):
    # needs to be rewriien to return location (i, j, k)
    instList = []
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            for k in range(world.shape[2]):
                if world[i, j, k].action_type == kind:
                    instList.append((i, j, k))
    return instList


def update_epsilon(epsilon, rate=0.99):
    # TODO, this goes down way too fast when turn is low
    # so, fast updatee mean that epsilon goes down faster than it should
    # if epsilon > 0.1:
    #    epsilon -= 1 / (turn)

    epsilon = max(0.1, rate * epsilon)
    return epsilon


def examine_replay(models, index, modelnum):

    image_r = models[modelnum].replay[index][0][0][0].numpy()
    image_g = models[modelnum].replay[index][0][0][1].numpy()
    image_b = models[modelnum].replay[index][0][0][2].numpy()

    image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    image_r = models[modelnum].replay[index][3][0][0].numpy()
    image_g = models[modelnum].replay[index][3][0][1].numpy()
    image_b = models[modelnum].replay[index][3][0][2].numpy()

    image2 = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    action = models[modelnum].replay[index][1]
    reward = models[modelnum].replay[index][2]
    done = models[modelnum].replay[index][4]

    return image, image2, (action, reward, done)


# look at a few replay games
def replay_games(numGames, modelNum, startingMem, models):
    for i in range(numGames):
        image, image2, memInfo = examine_replay(models, i + startingMem, modelNum)

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="Blues_r")
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap="Accent_r")

        print(memInfo)

        plt.show()


def examine_replay_memory(models, episode, index, modelnum):

    image_r = models[modelnum].replay.memory[episode][index][0][0][0].numpy()
    image_g = models[modelnum].replay.memory[episode][index][0][0][1].numpy()
    image_b = models[modelnum].replay.memory[episode][index][0][0][2].numpy()

    image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    image_r = models[modelnum].replay.memory[episode][index][3][0][0].numpy()
    image_g = models[modelnum].replay.memory[episode][index][3][0][1].numpy()
    image_b = models[modelnum].replay.memory[episode][index][3][0][2].numpy()

    image2 = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)

    action = models[modelnum].replay.memory[episode][index][1]
    reward = models[modelnum].replay.memory[episode][index][2]
    done = models[modelnum].replay.memory[episode][index][4]

    return image, image2, (action, reward, done)


# look at a few replay games


def replay_MemoryGames(models, modelNum, episode):
    epLength = len(models[modelNum].replay.memory[episode])
    for i in range(epLength):
        image, image2, memInfo = examine_replay_memory(models, episode, i, modelNum)

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="Blues_r")
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap="Accent_r")
        print(memInfo)
        plt.show()


def number_memories(modelNum, models):
    episodes = len(models[modelNum].replay.memory)
    print("there are ", episodes, " in the model replay buffer.")
    for e in range(episodes):
        epLength = len(models[0].replay.memory[e])
        print("Memory ", e, " is ", epLength, " long.")


def update_memories(env, expList, done, end_update=True):
    # update the reward and last state after all have moved
    # changed to holdObject to see if this fixes the failure of updating last memory
    for loc in expList:
        # location = (i, j, 0)
        #holdObject = env.world[loc]
        exp = env.world[loc].episode_memory[-1]
        lastdone = exp[1][4]
        if done == 1:
            lastdone = 1
        if end_update == False:
            exp = exp[0], (exp[1][0], exp[1][1], env.world[loc].reward, exp[1][3], lastdone)
        if end_update == True:
            input2 = env.pov(loc)
            exp = exp[0], (exp[1][0], exp[1][1], env.world[loc].reward, input2, lastdone)
        env.world[loc].episode_memory[-1] = exp
    return env.world


def transfer_world_memories(models, world, expList, extra_reward=True):
    # transfer the events from agent memory to model replay
    for loc in expList:
        # this moves the specific form of the replay memory into the model class
        # where it can be setup exactly for the model
        models[world[loc].policy].transfer_memories(world, loc, extra_reward=True)
    return models

def update_memories_rnn(env, expList, done, end_update=True):
    # update the reward and last state after all have moved
    # changed to holdObject to see if this fixes the failure of updating last memory
    for loc in expList:
        # location = (i, j, 0)
        #holdObject = env.world[loc]
        exp = env.world[loc].episode_memory[-1]
        lastdone = exp[1][4]
        if done == 1:
            lastdone = 1
        if end_update == False:
            exp = exp[0], (exp[1][0], exp[1][1], env.world[loc].reward, exp[1][3], lastdone, exp[1][5], exp[1][6])
        if end_update == True:
            input2 = env.pov(loc)
            exp = exp[0], (exp[1][0], exp[1][1], env.world[loc].reward, input2, lastdone, exp[1][5], exp[1][6])
        env.world[loc].episode_memory[-1] = exp
    return env.world
