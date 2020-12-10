# traffic-prediction

Paul Lecomte, Léo Hardy

Ce git résume le projet effectué lors du [cours de Machine Learning de M.Cristophe Cerisara](https://members.loria.fr/CCerisara/#courses/machine_learning/).

L'objectif est d'effectuer une prévision de traffic pour la ville d'Austin, à l'aide des données des radars de la ville, [disponibles sur ce challenge kaggle.](https://www.kaggle.com/vinayshanbhag/radar-traffic-data)

*Nte à l'intention de M.Cerisara : le nombre de commit de chacun de nous deux n'est pas forcément représentatif de nos travaux respectifs étant donné que nous avons travaillé et réfléchi à deux sur la même machine à de nombreux moments.*

## Etat de l'art et motivation de l'approche choisie

Nous nous sommes en premier lieu penchés sur plusieurs articles de recherche sur la prédiction de traffic.
Celui qui a le plus retenu notre attention est [celui-ci](https://www.researchgate.net/publication/333096680_Deep_Autoencoder_Neural_Networks_for_Short-Term_Traffic_Congestion_Prediction_of_Transportation_Networks), et propose de créer une représentation graphique de la ville avec une coloration présentant la densité de traffic.
Un autre article dont nous nous sommes inspirés est [celui-ci](https://www.researchgate.net/publication/340158853_Air_Pollution_Prediction_Using_Long_Short-Term_Memory_LSTM_and_Deep_Autoencoder_DAE_Models/link/5e7b59f7299bf1f3874008f0/download)

Bien que l'approche originale et les résultats présentés pertinents, nous avons dû abandonner cette piste trop ambitieuse (un combinaison de DAE et des LSTM) pour un projet de cette durée. Nous mettons de côté l'approche par DAE.


Nous nous sommes alors résolu à emprunter une approche plus familière pour nous, dans l'optique d'avoir des résultats à moyen terme, qui pourront enrichir notre réflexion.
Nous estimons donc qu'il est probablement plus judicieux de se contenir à la mise en place seulement d'un LSTM suivis de couches denses.


## Data processing

(Les graphiques suivant sont effectués avec la librairie matplotlib)

Une fois les données nettoyées (espaces inutiles avant le nom des lieux), nous nous sommes posé la question de la complétion de l'information fournie. En d'autres termes, combien de mesure manque-t-il par radar dans l'intervalle de temps proposé, pour chaque radar ?

Voici le pourcentage de valeurs manquantes pour chacun des radars :

radar | pourcentage de Nan
---|---
100 BLK S CONGRESS AVE (Congress Bridge) | 8.12%
1000 BLK W CESAR CHAVEZ ST (H&B Trail Underpass) | 8.41%
1612 BLK S LAMAR BLVD (Collier) | 23.7%
2021 BLK KINNEY AVE (NW 300ft NW of Lamar) | 81.0%
3201 BLK S LAMAR BLVD (BROKEN SPOKE) | 27.7%
400 BLK AZIE MORTON RD (South of Barton Springs Rd) | 22.0%
700 BLK E CESAR CHAVEZ ST | 8.62%
BURNET RD / PALM WAY (IBM DRIVEWAY) | 15.9%
BURNET RD / RUTLAND DR | 16.6%
CAPITAL OF TEXAS HWY / CEDAR ST | 36.3%
CAPITAL OF TEXAS HWY / LAKEWOOD DR | 11.1%
CAPITAL OF TEXAS HWY / WALSH TARLTON LN | 11.2%
CONGRESS AVE / JOHANNA ST (Fulmore Middle School) | 26.9%
LAMAR BLVD / MANCHACA RD | 10.7%
LAMAR BLVD / N LAMAR SB TO W 15TH RAMP | 24.1%
**LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)** | **4.35%**
LAMAR BLVD / SHOAL CREEK BLVD | 6.65%
LAMAR BLVD / ZENNIA ST | 33.6%

Nous avons ainsi décidé d'entraîner le réseau avec les données du radar nommé "LAMAR BLVD / SANDRA MURAIDA WAY (Lamar Bridge)", car le nombre de valeurs manquantes est faible.

Pour palier à ce léger manque, nous avons choisi de remplacer les valeurs manquantes par une régression linéaires entre les valeurs encadrant le laps de temps dont nous ignorons les mesures.
Par exemple, ci dessous la représentation du volume de voitures détectées au mois d'août 2017, avant d'extrapoler les données manquantes, et après.

![lamar_sept_2017_rough_datat](./images/lamar_sept_2017_rough_data.png)
![lamar_sept_2017_extrapolation](./images/lamar_sept_2017_extrapolation.png)

Dans un second temps, étant donné la répartition des données illustrées par l'histogramme ci-dessous, nous avons choisi de considérer un traffic par quart d'heure entre 0 et 1000, que nous avons ensuite normalisé entre 0 et 1.

![volume_repartition](./images/volume_repartition.png)
![volume_stats](./images/volume_stats.png)
