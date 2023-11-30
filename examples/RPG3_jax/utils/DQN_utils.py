from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from gem.game_utils import create_world, create_world_image
import matplotlib.animation as animation
import random
import pickle
# from gem.models.perception import agent_visualfield

from gem.utils import (
    find_instance,
    update_memories,
)
import torch


def create_video(
    models, world_size, num, env, filename="unnamed_video.gif", end_update=True
):
    fig = plt.figure()
    ims = []
    env.reset_env(world_size, world_size)
    done = 0
    for loc in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        # the parameter sets the length of the sequence for LSTM
        pov_size = (env.tile_size[0] * (env.world[loc].vision*2 + 1), env.tile_size[1] * (env.world[loc].vision*2 + 1))
        env.world[loc].init_replay(numberMemories=3, pov_size=pov_size, visual_depth=4)
        env.world[loc].init_rnn_state = None
    game_points = [0, 0]
    for _ in range(num):
        # image = create_world_image(env.world)
        # image = env.agent_visualfield(env.world, (0,0), k=int(world_size/2))
        image = env.plot_alt(0)
        im = plt.imshow(image, animated=True)
        ims.append([im])

        agentList = find_instance(env.world, "neural_network")
        random.shuffle(agentList)

        for loc in agentList:
            if env.world[loc].action_type == "neural_network":

                (
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    new_loc,
                    info,
                ) = env.step(models, loc, 0.2)

        env.world = update_memories(
            env,
            find_instance(env.world, "neural_network"),
            done,
            end_update=end_update,
        )

        # note that with the current setup, the world is not generating new wood and stone
        # we will need to consider where to add the transitions that do not have movement or neural networks
        regenList = find_instance(env.world, "deterministic")

        for loc in regenList:
            env.world = env.world[loc].transition(env.world, loc)

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename, writer="PillowWriter", fps=2)


def get_TD_error(
    models,
    policy,
    device,
    state,
    action,
    reward,
    next_state,
    done,
    gamma=0.95,
    offset=0.0001,
):
    Q1 = models[policy].model1(state.to(device))
    with torch.no_grad():
        Q2 = models[policy].model2(next_state.to(device))
    Y = reward + gamma * ((1 - done) * torch.max(Q2.detach(), dim=1)[0])

    X = Q1.detach()[0][action]

    error = torch.abs(Y - X).data.cpu().numpy()
    error = error + offset
    return error


def save_models(models, save_dir, filename):
    with open(save_dir + filename, "wb+") as fp:
        pickle.dump(models, fp)


def load_models(save_dir, filename):
    with open(save_dir + filename, "rb") as fp:
        model = pickle.load(fp)
    return model


def make_video(filename, save_dir, models, world_size, env, end_update=True):
    epoch = 10000
    for video_num in range(1):
        vfilename = (
            save_dir
            + filename
            + "_replayVid_"
            + str(epoch)
            + "_"
            + str(video_num)
            + ".gif"
        )
        create_video(
            models, world_size, 50, env, filename=vfilename, end_update=end_update
        )


def replay_view(memoryNum, agentNumber, env):
    agentList = find_instance(env.world, "neural_network")
    location = agentList[agentNumber]

    Obj = env.world[location]

    state = Obj.replay[memoryNum][0]
    next_state = Obj.replay[memoryNum][3]

    state_RGB = state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    image = make_lupton_rgb(
        state_RGB[:, :, 0], state_RGB[:, :, 1], state_RGB[:, :, 2], stretch=0.5
    )

    next_state_RGB = next_state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    imageNext = make_lupton_rgb(
        next_state_RGB[:, :, 0],
        next_state_RGB[:, :, 1],
        next_state_RGB[:, :, 2],
        stretch=0.5,
    )

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(imageNext)
    plt.show()
    print(Obj.replay[memoryNum][1], Obj.replay[memoryNum][2], Obj.replay[memoryNum][4])


def replay_view_model(memoryNum, modelNumber, models):
    state = models[modelNumber].replay[memoryNum][0]
    next_state = models[modelNumber].replay[memoryNum][3]

    state_RGB = state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    image = make_lupton_rgb(
        state_RGB[:, :, 0], state_RGB[:, :, 1], state_RGB[:, :, 2], stretch=0.5
    )

    next_state_RGB = next_state[:, -1, :, :, :].squeeze().permute(1, 2, 0).numpy()
    imageNext = make_lupton_rgb(
        next_state_RGB[:, :, 0],
        next_state_RGB[:, :, 1],
        next_state_RGB[:, :, 2],
        stretch=0.5,
    )

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(imageNext)
    plt.show()
    print(
        models[modelNumber].replay[memoryNum][1],
        models[modelNumber].replay[memoryNum][2],
        models[modelNumber].replay[memoryNum][4],
    )


def create_data(env, models, epochs, world_size):
    game_points = [0, 0]
    env.reset_env(world_size, world_size)
    for i, j, k in find_instance(env.world, "neural_network"):
        env.world[i, j, k].init_replay(3)
    for _ in range(epochs):
        game_points = env.step(models, game_points)
    return env


def create_video2(models, world_size, num, env, filename="unnamed_video.gif"):
    fig = plt.figure()
    ims = []
    env.reset_env(world_size, world_size, layers=2)
    done = 0
    for location in find_instance(env.world, "neural_network"):
        # reset the memories for all agents
        env.world[location].init_replay(3)
    game_points = [0, 0]
    for _ in range(num):
        image1 = create_world_image(env.world, layers=0)
        image2 = create_world_image(env.world, layers=1)

        for i in range(world_size):
            for j in range(world_size):
                R, G, B = image2[i, j]
                if R != 0 or G != 0 or B != 0:
                    image1[i, j][0] = R
                    image1[i, j][1] = G
                    image1[i, j][2] = B

        im = plt.imshow(image1, animated=True)
        ims.append([im])

        agentList = find_instance(env.world, "neural_network")
        random.shuffle(agentList)

        for loc in agentList:
            if env.world[loc].action_type == "neural_network":

                (
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    new_loc,
                    info,
                ) = env.step(models, loc, 0.2)

        regenList = find_instance(env.world, "deterministic")

        for loc in regenList:
            env.world = env.world[loc].transition(env.world, loc)

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename, writer="PillowWriter", fps=2)


def make_video2(filename, save_dir, models, world_size, env):
    epoch = 10000
    for video_num in range(5):
        vfilename = (
            save_dir
            + filename
            + "_replayVid_"
            + str(epoch)
            + "_"
            + str(video_num)
            + ".gif"
        )
        create_video2(models, world_size, 100, env, filename=vfilename)
