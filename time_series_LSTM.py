import torch
import torch.nn as nn


"""

Création du model d'un réseau convolutionel avec deux convolutions et un feedforward (un enchaînement de couches denses).

"""


class LSTM(nn.Module):
    # On a qu'une seule variable d'entrée qui est le nombre de véhicules détectés sur un radar

    def __init__(self, input_size=1, hidden_layer_size=75, sortie=1):

        super().__init__()
        self.couche_cachee = hidden_layer_size
        # définition du module lstm
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        # définition de le 3 couches denses en sorties du module LSTM
        self.fc1 = nn.Linear(couche_cachee, 75)
        self.fc2 = nn.Linear(75, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward():
        # Après chaque couche, on utilise la fonction d'activation ReLu
        x = F.relu(self.lstm(x))
        x = F.relu(self.fc1(x))
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


# Je mets le nombre d'epoch élevé quitte à me mettre en situation de surapprentissage
for epoch in range(60):

# A faire en fonction du data processing réalisé

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
