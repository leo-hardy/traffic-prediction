import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

"""

Implementation de la sliding window

"""
#lecture des données
data = pd.read_csv("../LAMAR BLVD.csv")


#normalisation des donnees
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(data["LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)"].to_numpy().reshape(-1, 1))
train_data_normalized = train_data_normalized.reshape(1,-1)[0][:400]


def createur_vecteur(sequence, pas):
    seq=[]
    for i in range(0,len(sequence)-pas-1):
        seq.append((torch.FloatTensor(np.array([sequence[j] for j in range(i,i+pas)])),torch.FloatTensor(np.array(sequence[i+pas]))))
    return seq


print(createur_vecteur(train_data_normalized, 40))


"""

Création du model d'un réseau convolutionel avec deux convolutions et un feedforward (un enchaînement de couches denses).

"""


class LSTM(nn.Module):
    # On a qu'une seule variable d'entrée qui est le nombre de véhicules détectés sur un radar

    def __init__(self, input_size=1, hidden_layer_size=75, output_size=1):

        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        # définition du module lstm
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        # définition de le 3 couches denses en sorties du module LSTM
        self.fc1 = nn.Linear(self.hidden_layer_size, 75)
        self.fc2 = nn.Linear(75, 25)
        self.fc3 = nn.Linear(25, 1)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))


    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1), self.hidden_cell)
        # Après chaque couche, on utilise la fonction d'activation ReLu
        x = F.relu(self.fc1(lstm_out[-1].view(len(input), -1)))
        x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return 1




net = LSTM()
# Nous choisissons d'utiliser l'erreur quadratique moyenne comme criterion et l'optimize Adam (choix relativement arbitraire..)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())


"""

Mise en place de la boucle d'apprentissage

"""





"""

# Je mets le nombre d'epoch élevé quitte à me mettre en situation de surapprentissage
for epoch in range(60):

# A faire en fonction du data processing réalisé

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
"""

'''
### THE LOOP

# Lists for visualization of loss and accuracy 
loss_list = []
iteration_list = []
errors_test_set_list = []

for epoch in range(num_epochs):
    for traffic_previous, traffic_real in train_loader:

        # Transfering images and labels to GPU if available
        traffic_previous, traffic_real = traffic_previous.to(device), traffic_real.to(device)
        
        # it can change for the last batch !
        batch_size = traffic_previous.size()[0]
        
        train = Variable( traffic_previous.view( batch_size, 1, ?, ? ) )
        traffic_real = Variable( traffic_real )
        
        # Forward pass
        outputs = torch.round( 20* net( train ) ) /20
        
        loss = criterion( outputs, traffic_real )
        
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()
        
        #Propagating the error backward
        loss.backward()
        
        # Optimizing the parameters
        optimizer.step()
    
        count += 1
    
    # Testing the model
    
        if not ( count % 100 ):
            total = 0
            err = 0
            
            test_count = 0
        
            for traffic_previous, traffic_real in test_loader:
                
                traffic_previous, traffic_real = traffic_previous.to(device), traffic_real.to(device)
                labels_list.append( traffic_real )
                
                # it can change for the last batch !
                batch_size = traffic_previous.size()[0]
                
                test = Variable( traffic_previous.view(batch_size, 1, ?, ?) )
                # we want int values for sales but we got [0, 1] values in nn
                outputs = torch.round( net( train ) )
            
                # root mean square error
                err += criterion( )
                
                #print("err", err)
                total += len( traffic_real )
                

            
            errors_test_set = np.true_divide( err.detach().numpy() , total) ?
            loss_list.append(loss.data)
            iteration_list.append(count)
            errors_test_set_list.append(errors_test_set)
        
        if not (count % 100):
            print("Iteration: {}, Loss: {}, errors_test_set: {} /(item, shop)".format(count, loss.data, errors_test_set))



plt.plot(iteration_list, errors_test_set_list)
plt.xlabel("No. of Iteration")
plt.ylabel("errors_test_set")
plt.title("Iterations vs errors_test_set, lr=%s, batch size=%s, %s epochs"%(learning_rate, batch_size, num_epochs))
plt.show()
'''