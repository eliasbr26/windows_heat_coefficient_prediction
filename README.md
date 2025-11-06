# Windows Heat Coefficient Prediction

Un projet de machine learning pour prédire le coefficient thermique / de transfert de chaleur de fenêtres (valeur U, conductance thermique ou autre métrique de performance thermique) à partir de caractéristiques disponibles (géométrie, matériaux, conditions environnementales, etc.).

Ce dépôt contient le code, les notebooks et les scripts nécessaires pour entraîner, évaluer et déployer des modèles de régression qui estiment la performance thermique des fenêtres.

## Table des matières
- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Structure du dépôt](#structure-du-dépôt)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Jeux de données](#jeux-de-données)
- [Usage](#usage)
  - [Exploration avec notebook](#exploration-avec-notebook)
  - [Entraînement](#entraînement)
  - [Évaluation](#évaluation)
  - [Prédiction / Inference](#prédiction--inference)
- [Approche modèle](#approche-modèle)
- [Métriques d'évaluation](#métriques-dévaluation)
- [Reproductibilité](#reproductibilité)
- [Contribuer](#contribuer)
- [Licence](#licence)
- [Contact](#contact)

## Aperçu
L'objectif est de fournir une pipeline reproductible permettant :
- de préparer et valider des jeux de données sur les fenêtres,
- d'entraîner plusieurs modèles de régression (baseline et modèles avancés),
- d'évaluer et comparer leurs performances,
- et d'exposer un point d'entrée (script ou API) pour faire des prédictions sur de nouvelles fenêtres.

## Fonctionnalités
- Pipeline de prétraitement (nettoyage, encodages, normalisation)
- Entraînement configurable (fichiers de configuration / arguments CLI)
- Évaluation et visualisations des résultats
- Notebooks d'exploration et de démonstration
- Enregistrements de modèles et gestion des artefacts

## Structure du dépôt
(Exemple attendu — adaptez si différent)
- data/                      — jeux de données (ne pas committer les données sensibles)
- notebooks/                 — notebooks d'exploration et d'expérimentation
- src/                       — code source (prétraitement, modèles, entraînement, utils)
- configs/                   — configurations pour entraînement/expérimentation
- models/                    — modèles entraînés / checkpoints
- results/                   — métriques, graphiques et rapports
- requirements.txt           — dépendances Python
- README.md

## Prérequis
- Python 3.8+
- pip (ou conda)
- Espace disque suffisant pour les jeux de données et les checkpoints

## Installation
1. Cloner le dépôt :
   ```
   git clone https://github.com/eliasbr26/windows_heat_coefficient_prediction.git
   cd windows_heat_coefficient_prediction
   ```

2. Créer un environnement virtuel et activer :
   - Avec venv :
     ```
     python -m venv .venv
     source .venv/bin/activate  # Linux / macOS
     .\.venv\Scripts\activate   # Windows
     ```
   - Ou avec conda :
     ```
     conda create -n whcp python=3.9
     conda activate whcp
     ```

3. Installer les dépendances :
   ```
   pip install -r requirements.txt
   ```

## Jeux de données
- Structure attendue (exemple) : un fichier CSV `data/windows.csv` contenant une ligne par fenêtre avec colonnes telles que :
  - id, width, height, glazing_type, frame_material, insulation_thickness, indoor_temp, outdoor_temp, measured_coefficient
- Le script de prétraitement (src/data_preprocessing.py) prend en entrée le CSV brut et produit des jeux train/val/test.
- IMPORTANT : ne commitez pas les jeux de données sensibles dans le dépôt public. Utilisez `.gitignore` pour exclure `data/` si nécessaire.

## Usage

### Exploration avec notebook
- Ouvrir et exécuter les notebooks dans `notebooks/` pour comprendre les données et tester des pipelines :
  ```
  jupyter lab
  ```
  ou
  ```
  jupyter notebook
  ```
  Puis ouvrir `notebooks/01_exploration.ipynb` (nom d'exemple).

### Entraînement
- Exemple de commande CLI (adapter selon les scripts présents) :
  ```
  python src/train.py --config configs/train.yaml
  ```
- Paramètres typiques :
  - chemin vers les données
  - hyperparamètres (learning rate, epochs, batch size)
  - chemin de sortie pour les modèles

### Évaluation
- Après entraînement, exécuter :
  ```
  python src/evaluate.py --model models/latest_model.pkl --test data/test.csv
  ```
- Génère un rapport avec métriques et courbes (ex : prédictions vs vrai).

### Prédiction / Inference
- Exemple d'utilisation pour prédire un seul échantillon :
  ```
  python src/predict.py --model models/latest_model.pkl --input '{"width":1.2,"height":1.5,"glazing_type":"double","frame_material":"uPVC",...}'
  ```
- Ou charger le modèle dans un notebook pour des prédictions en batch.

## Approche modèle
- Problème formulé comme une régression supervisée.
- Baselines recommandées :
  - Régression linéaire / Ridge / Lasso
  - Random Forest Regressor
  - XGBoost / LightGBM
  - Un réseau neuronal simple si jeu de données volumineux
- Comparer modèles via validation croisée et jeu de test dédié.

## Métriques d'évaluation
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (coefficient de détermination)
- Visualisations : courbe prédiction vs réalité, histogramme des erreurs

## Reproductibilité
- Fixer le seed global dans les scripts (numpy, random, torch) pour rendre les expériences reproductibles.
- Enregistrer la configuration complète (fichier config, hash du commit) avec chaque artefact.

## Contribuer
- Forkez le dépôt, créez une branche dédiée (feature/xxx ou fix/yyy).
- Ouvez une pull request avec description claire des modifications.
- Ajoutez des tests pour tout comportement nouveau ou modifié.
- Respectez le style de code (PEP8). Linter/config peut être ajouté au besoin.

## Licence
Précisez la licence du projet (ex : MIT). Si aucun fichier LICENSE n'est présent, ajoutez-en un avant de publier.

## Contact
Pour toute question, contactez l'auteur / mainteneur : @eliasbr26 (GitHub).

---

Notes :
- Adaptez les noms de fichiers et les commandes à la structure réelle du dépôt si elle diffère.
- Pensez à ajouter un fichier `requirements.txt` ou `environment.yml` et un `LICENSE`.
