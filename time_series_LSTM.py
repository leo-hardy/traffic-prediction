import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time
from model import LSTM
from createur_vecteur import createur_vecteur

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_time = time.time()

### STYLE DEFINITION ###
plt.style.use('ggplot')

### MISE EN PLACE DES DONNEES ###

# Lecture des données
my_radar = 'LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)'
data = pd.read_csv("../LAMAR BLVD.csv")

data_lower_bound = 40000
data_upper_bound = 50000

# Normalisation des donnees entre -1 et 1 en utilisant la fonction MinMaxScaler de la librairie sklearn
scaler = MinMaxScaler( feature_range=(-1, 1) )
data_normalized = scaler.fit_transform( data[ my_radar ].to_numpy().reshape(-1, 1) )
data_normalized_for_processing = data_normalized.reshape(1,-1)[0][data_lower_bound :data_upper_bound]

# length of the window for training, it is the number of previous quarter-hours from which the net learns
window_length = 4*24 # for one day
batch_size = 1
print( 'hyperparameters : window_length = %s hours' % (window_length/4) )

radar_sequences = createur_vecteur( data_normalized_for_processing, window_length )
# Obtention du training set et du test set a partir d'une fonction de sklearn
train_seq, test_seq = train_test_split( radar_sequences, test_size = 1000)

data_time = time.time()
print('training and testing set preparation took %s seconds' % (data_time-start_time) )




### MODELE ###

# import du modèle
net = LSTM()


# Nous choisissons d'utiliser l'erreur quadratique moyenne comme criterion et l'optimize Adam (choix relativement arbitraire..)
criterion = nn.MSELoss()

learning_rate = 0.001

optimizer = torch.optim.Adam( net.parameters(), lr=learning_rate )

model_time = time.time()
print('Model creation took %s seconds' % (model_time-data_time) )



### BOUCLE D'APPRENTISSAGE ###

num_epochs = 4

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

        # Initializing a gradient as 0 
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

        # Testing the model every *checkpoint* iterations
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

### PRINT RESULTS ###

print('average test overall all test set took %s seconds' % ( sum(duration_test_list)/len(duration_test_list) ) )
plt.plot( iteration_list, loss_list, color='#32378D', label='Train loss' )
plt.plot( iteration_list, errors_test_set_list, color='#FF761A', label='Test loss'  )
plt.legend(loc="upper right")
plt.xlabel( "No. of Iteration" )
plt.ylabel( "errors" )
plt.title( "Iterations vs errors_test/train_set, %s epochs, window of %s 1/4 hours, lr=%s" % ( num_epochs, window_length, learning_rate ))
plt.show()


print("TOTAL took %s seconds" % (time.time() - start_time))
