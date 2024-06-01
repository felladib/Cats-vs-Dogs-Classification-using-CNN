# Classification de Chiens et Chats avec Réseaux de Neurones Convolutionnels (CNN)

Ce projet vise à classer les images de chiens et de chats en utilisant un Réseau de Neurones Convolutionnels (CNN) construit avec TensorFlow et Keras. Le jeu de données utilisé est issu de la [compétition Kaggle](https://www.kaggle.com/c/dogs-vs-cats).

## Objectif

L'objectif principal de ce projet est de créer un modèle capable de distinguer entre les images de chiens et de chats avec une précision élevée.

## Table of Contents

- [Dataset](#Dataset)
- [Installation](#Installation)
- [Data Preparation](#Préparation-des-Données)
- [Model Architecture](#Architecture-du-Modèle)
- [Training](#Entraînement)
- [Evaluation](#Évaluation)
- [Results](#Résultats)


## Dataset

The dataset consists of images of cats and dogs. The data is split into training and testing sets:

- `train.zip`: Contains 25,000 labeled images of cats and dogs.
- `test1.zip`: Contains 12,500 unlabeled images for prediction.

## Installation


Pour exécuter ce projet localement, suivez ces étapes :

1. Assurez-vous d'avoir Python 3.x installé sur votre système.
2. Installez les dépendances en exécutant `pip install -r requirements.txt`.

- TensorFlow
- Keras
- NumPy
- Pandas
- Pillow
- Matplotlib
- scikit-learn



## Préparation-des-Données

1. Téléchargez le jeu de données à partir du lien fourni dans la section "Dataset".
2. Extrayez les fichiers zip dans le dossier approprié.
3. Utilisez les scripts fournis pour prétraiter les données et les préparer pour l'entraînement.

## Architecture-du-Modèle

Le modèle utilisé est un réseau de neurones convolutif (CNN) avec les caractéristiques suivantes :

- Trois couches de convolution avec fonction d'activation ReLU.
- Normalisation par lots après chaque couche de convolution.
- Max pooling pour réduire les dimensions spatiales.
- Couches de dropout pour la régularisation.
- Une couche dense finale avec une activation softmax pour la classification.

## Entraînement

Pour entraîner le modèle, exécutez le script `train.py` en spécifiant les paramètres appropriés. Le modèle sera entraîné sur les données d'entraînement et évalué sur les données de validation.

## Évaluation

Après l'entraînement, le modèle sera évalué sur les données de test pour évaluer sa performance. Les résultats seront affichés dans la console et enregistrés dans un fichier de journal.

## Résultats

Les résultats de l'entraînement et de l'évaluation seront disponibles dans le fichier de journal généré après l'exécution du script.

## Contributions

Les contributions sont les bienvenues ! Si vous avez des suggestions d'amélioration ou des correctifs à apporter, veuillez ouvrir une issue ou soumettre une pull request.



