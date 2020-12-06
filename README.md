# traffic-prediction


## Etat de l'art et motivation de l'approche choisie

Nous nous sommes en premier lieu penchés sur plusieurs articles de recherche sur la prédiction de traffic.
Celui qui a le plus retenu notre attention est [celui-ci](https://www.researchgate.net/publication/333096680_Deep_Autoencoder_Neural_Networks_for_Short-Term_Traffic_Congestion_Prediction_of_Transportation_Networks), et propose de créer une représentation graphique de la ville avec une coloration présentant la densité de traffic.
Un autre article dont nous nous sommes inspirés est [celui-ci](https://www.researchgate.net/publication/340158853_Air_Pollution_Prediction_Using_Long_Short-Term_Memory_LSTM_and_Deep_Autoencoder_DAE_Models/link/5e7b59f7299bf1f3874008f0/download)

Bien que l'approche originale et les résultats présentés pertinents, nous avons dû abandonner cette piste trop ambitieuse (un combinaison de DAE et des LSTM) pour un projet de cette durée. Nous mettons de côté l'approche par DAE.


Nous nous sommes alors résolu à emprunter une approche plus familière pour nous, dans l'optique d'avoir des résultats à moyen terme, qui pourront enrichir notre réflexion.
Nous estimons donc qu'il est probablement plus judicieux de se contenir à la mise en place seulement d'un LSTM suivis de couches denses.
