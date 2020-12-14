import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.utils import shuffle
import time
from model import LSTM

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
# Lecture des données
start_time = time.time()

data = pd.read_csv("../LAMAR BLVD.csv")

data_lower_bound = 50000
data_upper_bound = 55000

# Normalisation des donnees entre -1 et 1 en utilisant la fonction MinMaxScaler de la librairie sklearn
scaler = MinMaxScaler( feature_range=(-1, 1) )
data_normalized = scaler.fit_transform( data["LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)"].to_numpy().reshape(-1, 1) )
data_normalized_for_processing = data_normalized.reshape(1,-1)[0][data_lower_bound :data_upper_bound]

# length of the window for training, it is the number of previous quarter-hours from which the net learns
window_length = 4*24 # for one day
batch_size = 1
print( 'hyperparameters : window_length = %s hours' % (window_length/4) )


# Cette fonction permet de creer une fenetre de valeur avec le label associé.
def createur_vecteur(sequence, pas):
    seq=[]
    for i in range(0,len(sequence)-pas-1):
        seq.append((torch.FloatTensor(np.array([sequence[j] for j in range(i,i+pas)])),torch.FloatTensor(np.array(sequence[i+pas]))))
    return seq


radar_sequences = createur_vecteur( data_normalized_for_processing, window_length )
# Obtention du training set et du test set a partir d'une fonction de sklearn
train_seq, test_seq = train_test_split( radar_sequences, test_size=50)



data_time = time.time()
print('training and testing set preparation took %s seconds' % (data_time-start_time) )


net = LSTM()


# Nous choisissons d'utiliser l'erreur quadratique moyenne comme criterion et l'optimize Adam (choix relativement arbitraire..)
criterion = nn.MSELoss()


learning_rate = 0.001

optimizer = torch.optim.Adam( net.parameters(), lr=learning_rate )

model_time = time.time()
print('Model creation took %s seconds' % (model_time-data_time) )


"""

Mise en place de la boucle d'apprentissage

"""
num_epochs = 1


# Lists for visualization of loss and accuracy
loss_list = []
iteration_list = []
errors_test_set_list = []
duration_test_list = []
loss_denormalized_sum = 0

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
        loss_denormalized_sum += np.sqrt( criterion( torch.FloatTensor(scaler.inverse_transform(traffic_predicted.detach().numpy() )[0]), torch.FloatTensor(scaler.inverse_transform( traffic_real.reshape(-1, 1) )[0]) ) )
        # Propagating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        count += 1

        # Testing the model every 100 iterations
        checkpoint = 200
        if not ( count % checkpoint ):
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
            loss_list.append( np.true_divide( loss_denormalized_sum.detach().numpy() , checkpoint ) )
            loss_denormalized_sum = 0



print('average test overall all test set took %s seconds' % ( sum(duration_test_list)/len(duration_test_list) ) )
plt.plot( iteration_list, loss_list, color='r', label='Train loss' )
plt.plot( iteration_list, errors_test_set_list, color='b', label='Test loss'  )
plt.legend(loc="upper right")
plt.xlabel( "No. of Iteration" )
plt.ylabel( "errors_test_set" )
plt.title( "Iterations vs errors_test_set, batch size=%s, %s epochs, window of %s 1/4 hours, lr=%s" % ( batch_size, num_epochs, window_length, learning_rate ))
plt.show()


# Plot the forcasting of car flow after the chosen data:
nb_days_predicted= 4
data_realized = data["LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)"][data_upper_bound : data_upper_bound+4*24*nb_days_predicted].to_numpy()
data_normalized_to_be_predicted = data_normalized.reshape(1,-1)[0][data_upper_bound : data_upper_bound+4*24*nb_days_predicted]

radar_sequences_to_be_predicted = createur_vecteur( data_normalized_to_be_predicted, window_length )

data_predicted = []
# Cette fenetre contient les valeurs prédites par le réseau
current_window = radar_sequences_to_be_predicted[0][0]
counter = 0
for traffic_previous, traffic_real in radar_sequences_to_be_predicted:
    traffic_previous, traffic_real = traffic_previous.to(device), traffic_real.to(device)
    traffic_predi = net( traffic_previous ).detach()[0]
    if counter < len(radar_sequences_to_be_predicted[0][0]):
        current_window = torch.cat((current_window[1:],traffic_predi),0)
    else:
        current_window = torch.cat((current_window,traffic_predi),0)

    # we want int values for sales but we got [0, 1] values in nn
    traffic_predicted = scaler.inverse_transform( net( current_window ).detach().numpy() )
    data_predicted.append(traffic_predicted[0][0])
    counter =+ 1


plt.plot([i for i in range(len(data_predicted))], data_predicted, color='r', label='Predicted' )
plt.plot([i for i in range(len(data_realized))],data_realized, color='b', label='Realized' )
plt.legend(loc="upper right")
plt.show()

print("TOTAL took %s seconds" % (time.time() - start_time))
