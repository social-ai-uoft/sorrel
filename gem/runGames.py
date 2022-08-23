#!/usr/bin/env python
# coding: utf-8

# from gemGame import runCombinedTraining, moreTraining
from gemGame_experimental import train_wolf_gem, save_models, load_models

# RUNNING THE MODELS BELOW

save_dir = "/Users/wil/Dropbox/Mac/Documents/gemOutput_experimental/"

models = train_wolf_gem(60000)
save_models(models, save_dir, "test", 10)
