
import torch
import torch.nn as nn


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

        # définition de la couche dense en sortie du module LSTM
        self.fc1 = nn.Linear( in_features=self.hidden_layer_size, out_features=1)


        self.hidden_cell = (
            torch.zeros( 1, 1, self.hidden_layer_size ),
            torch.zeros( 1, 1, self.hidden_layer_size )
        )


    def forward(self, input):

        lstm_out, self.hidden_cell = self.lstm( input.view( len(input), 1, -1 ), self.hidden_cell )
        # Après chaque couche, on utilise la fonction d'activation ReLu
        # on extrait la derniere couche h_t_finale.
        x = lstm_out[-1].view( -1, self.hidden_layer_size )
        x = self.fc1( x )
        return x

    # Nous remettons à zéro les couches cachées à chaque itération lors de l'apprentissage (Stateless Lstm)
    def reset_hidden_state(self):
        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size)
        )

