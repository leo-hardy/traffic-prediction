import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import time

from model import LSTM


model = LSTM()
model.load_state_dict( torch.load( '../model_saved.pt' ) )
model.eval()



