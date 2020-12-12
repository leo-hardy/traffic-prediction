import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
HYPERPARAMETERS:

taille de la sliding window
batch_size (minibatch ?)
num_epochs
Tip 3: Tune batch size and learning rate after tuning all other hyperparameters.
"""



"""

Implementation de la sliding window

"""
start_time = time.time()

#lecture des données
data = pd.read_csv("../LAMAR BLVD.csv")


#normalisation des donnees
scaler = MinMaxScaler( feature_range=(-1, 1) )
train_data_normalized = scaler.fit_transform( data["LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)"].to_numpy().reshape(-1, 1) )
train_data_normalized = train_data_normalized.reshape(1,-1)[0][:5000]

# length of the window for training, it is the number of previous quarter-hours from which the net learns
window_length = 4*2
batch_size = 1
print( 'hyperparameters : window_length = %s hours' % (window_length/4) )

def createur_vecteur(sequence, pas):
    seq=[]
    for i in range(0,len(sequence)-pas-1):
        seq.append((torch.FloatTensor(np.array([sequence[j] for j in range(i,i+pas)])),torch.FloatTensor(np.array(sequence[i+pas]))))
    return seq


radar_sequences = createur_vecteur( train_data_normalized, window_length )
train_seq, test_seq = train_test_split( radar_sequences, test_size=300 )

data_time = time.time()
print('training and testing set preparation took %s seconds' % (data_time-start_time) )

"""

Création du model d'un réseau convolutionel avec deux convolutions et un feedforward (un enchaînement de couches denses).

"""


class LSTM(nn.Module):
    # On a qu'une seule variable d'entrée qui est le nombre de véhicules détectés sur un radar

    def __init__(self, input_size=1, hidden_layer_size=80, output_size=1):

        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        # définition du module lstm à un seul bloc
        self.lstm = nn.LSTM( input_size=input_size, num_layers=1, hidden_size=self.hidden_layer_size )
        # définition de le 3 couches denses en sorties du module LSTM
        self.fc1 = nn.Linear( in_features=self.hidden_layer_size, out_features=64 )
        self.fc2 = nn.Linear( in_features=64, out_features=32)
        self.fc3 = nn.Linear( in_features=32, out_features=1)

        self.hidden_cell = ( 
            torch.zeros( 1, 1, self.hidden_layer_size ),
            torch.zeros( 1, 1, self.hidden_layer_size )
        )


    def forward(self, input):

        lstm_out, self.hidden_cell = self.lstm( input.view( len(input), 1, -1 ), self.hidden_cell )
        # Après chaque couche, on utilise la fonction d'activation ReLu
        # on extrait la derniere couche h_t_finale.

        x = lstm_out[-1].view( -1, self.hidden_layer_size )
        x = F.relu( self.fc1( x )) # POURQUOI UN RELU ICI ?
        x = F.relu( self.fc2( x )) # POURQUOI UN RELU ICI ?
        x = self.fc3( x ) # POURQUOI PAS UN RELU ICI ?
        return x

    def reset_hidden_state(self):
        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size)
        )



net = LSTM()
# Nous choisissons d'utiliser l'erreur quadratique moyenne comme criterion et l'optimize Adam (choix relativement arbitraire..)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam( net.parameters() )

model_time = time.time()
print('model creation took %s seconds' % (model_time-data_time) )


"""

Mise en place de la boucle d'apprentissage

"""
num_epochs = 1
""" Besoin de desordonner les données du train set """

# Lists for visualization of loss and accuracy
loss_list = []
iteration_list = []
errors_test_set_list = []
duration_test_list = []

count = 0
for epoch in range( num_epochs ):
    for traffic_previous, traffic_real in train_seq:

        # Transfering images and labels to GPU if available
        traffic_previous, traffic_real = traffic_previous.to(device), traffic_real.to(device)

        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()
        net.reset_hidden_state()

        traffic_predicted = net( traffic_previous )

        loss = criterion( traffic_predicted[0][0], traffic_real )

        #Propagating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        count += 1

        # Testing the model
        if not ( count % 100 ):
            t1 = time.time()
            err = 0

            for traffic_previous, traffic_real in test_seq:

                traffic_previous, traffic_real = traffic_previous.to(device), traffic_real.to(device)

                # we want int values for sales but we got [0, 1] values in nn
                traffic_real = scaler.inverse_transform( traffic_real.reshape(-1, 1) )
                traffic_predicted = scaler.inverse_transform( net( traffic_previous ).detach().numpy() )

                # root mean square error
                err += np.sqrt( criterion( torch.FloatTensor(traffic_predicted[0]), torch.FloatTensor(traffic_real[0]) ) )

            errors_test_set = np.true_divide( err.detach().numpy() , len( test_seq ))
            iteration_list.append( count )
            errors_test_set_list.append( errors_test_set )
            duration_test_list.append( time.time() - t1 )

            print("Iteration: {}, errors_test_set: {} /(quarter hour)".format( count, errors_test_set ))

print('average test overall all test set took %s seconds' % ( sum(duration_test_list)/len(duration_test_list) ) )

plt.plot( iteration_list, errors_test_set_list )
plt.xlabel( "No. of Iteration" )
plt.ylabel( "errors_test_set" )
plt.title( "Iterations vs errors_test_set, batch size=%s, %s epochs, window of %s 1/4 hours" % ( batch_size, num_epochs, window_length ))
plt.show()
