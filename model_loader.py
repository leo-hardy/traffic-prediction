import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from createur_vecteur import createur_vecteur
from model import LSTM

### STYLE DEFINITION ###
plt.style.use('ggplot')

### VARIABLES ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

window_length = 4*24 # for one day
data_lower_bound = 50000
data_upper_bound = 55000

data = pd.read_csv("../LAMAR BLVD.csv")

### LOAD MODEL ###
model = LSTM()
model.load_state_dict( torch.load( './model_saved.pt' ) )
model.eval()

### PREDICT ON NEW DATA ###
scaler = MinMaxScaler( feature_range=(-1, 1) )
data_normalized = scaler.fit_transform( data["LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)"].to_numpy().reshape(-1, 1) )

# Plot the forcasting of car flow after the chosen data:
nb_days_predicted = 10
data_realized = data["LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)"][data_upper_bound : data_upper_bound+4*24*nb_days_predicted].to_numpy()
data_normalized_to_be_predicted = data_normalized.reshape(1,-1)[0][data_upper_bound : data_upper_bound+4*24*nb_days_predicted]

radar_sequences_to_be_predicted = createur_vecteur( data_normalized_to_be_predicted, window_length )

data_predicted = []
# Cette fenetre contient les valeurs prédites par le réseau
current_window = radar_sequences_to_be_predicted[0][0]
counter = 0
for traffic_previous, traffic_real in radar_sequences_to_be_predicted:
    traffic_previous, traffic_real = traffic_previous.to(device), traffic_real.to(device)
    traffic_predi = model( traffic_previous ).detach()[0]
    if counter < len(radar_sequences_to_be_predicted[0][0]):
        current_window = torch.cat((current_window[1:],traffic_predi), 0)
    else:
        current_window = torch.cat((current_window,traffic_predi), 0)

    # we want int values for sales but we got [0, 1] values in nn
    traffic_predicted = scaler.inverse_transform( model( current_window ).detach().numpy() )
    data_predicted.append(traffic_predicted[0][0])
    counter =+ 1


### PLOT ###

plt.plot([i for i in range(len(data_predicted))],data_realized[0:len(data_predicted)], color='#32378D', label='Realized' )
plt.plot([i for i in range(len(data_predicted))], data_predicted, color='#FF761A', label='Predicted' )
plt.legend(loc="upper right")
plt.xlabel( "No. of quater hours" )
plt.ylabel( "Number of cars" )
plt.title("Predictions Vs Reality on {} days".format(str(nb_days_predicted)))
plt.show()
