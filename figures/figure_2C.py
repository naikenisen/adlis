""" 
A) Prédiction du pourcentage de SC par patients : le scripte reprendre la pipeline utilisée par
app/app.py, pour la détection suivie de la classification des SC et SN, sur chaques images des sous-dossiers de
dataset/test_externe. Pour chaque sous-dossiers de dataset/test_externe il détermine un pourcentrage
de sidéroblastes en couronnes avec la formule suivant ((SC/SC+SN)*100). 
Le scripte crée "dataset/inference-test-externe.csv" avec deux colonnes : "id" qui correspond au sous-dossiers
et "prediction" qui correspond au pourcentrage de sidéroblastes en couronnes.

B) Création du Bland-Altman plot et régression linéaire : le scripte reprendre 
"dataset/test-externe.csv" et "dataset/inference-test-externe.csv" pour produire 
un Bland-Altman plot et une regression linéaire pour comparer les "prediction" de 
"dataset/inference-test-externe.csv" aux "valeur" de "dataset/test-externe.csv". 
La correspondance se fera entre "id" de "dataset/test-externe.csv" et "id" de 
"dataset/inference-test-externe.csv".

"""
