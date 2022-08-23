#!/usr/bin/env python
# coding: utf-8

# from gemGame import runCombinedTraining, moreTraining
from gemGame_experimental_LSTM import train_wolf_gem, save_models, load_models

# RUNNING THE MODELS BELOW

save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"

models = train_wolf_gem(1000)
save_models(models, save_dir, "lstm_build_replay_memories", 3)
