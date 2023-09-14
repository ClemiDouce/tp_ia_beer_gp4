# Rapport de Projet - Brew

## Nettoyage des données

On a d'abord nettoyé les données du fichiers d'import.

On a supprimé les colonnes qui comportait trop de données manquantes. On a aussi supprimé les données qui semblait etre des erreurs, avec des valeurs trop haute ou trop basse.

On a utilisé la technique de Hot One sur les colonnes avec peu de valeurs différents sous forme de chaine de charactere pour pouvoir mieux les exploiter en suivant.

On a aussi fait une matrice de corrélation pour voir les corrélation entre les différentes colonnes
![Matrice de correlation de nos données](./matrice_correlation.jpg)

On a constaté une corrélation forte entre la colonne ABV et OG
![Correlation ABV-OG](./ABV-OG.png)

On a constaté aucune correlation entre la colonne ABV et la colonne BoilTime
![](./ABV-BoilTime.png)
