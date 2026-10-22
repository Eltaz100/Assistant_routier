# Assistant Routier
### Détection de panneaux routiers avec l'intelligence artificielle

## 📌 Présentation du projet

RoadSign Assistant est un projet de vision par ordinateur qui permet de détecter et de reconnaître plusieurs catégories de panneaux routiers dans une image grâce au modèle YOLO et au langage Python.

Ce projet couvre les principales étapes d'un pipeline de détection d'objets : préparation des données, entraînement du modèle, évaluation des performances et prédiction sur de nouvelles images.

## ✨ Fonctionnalités

- Détection de plusieurs panneaux routiers dans une même image.
- Encadrement des panneaux détectés avec des boîtes de détection.
- Affichage des classes reconnues et des scores de confiance.
- Évaluation du modèle à l'aide de métriques de performance.
- Interface interactive avec Streamlit prévue pour la démonstration.

## 🛠️ Technologies utilisées

- **Python** : langage de programmation.
- **YOLO et Ultralytics** : détection d'objets.
- **PyTorch** : framework d'apprentissage automatique.
- **Roboflow** : préparation et exportation du dataset.
- **Google Colab** : entraînement du modèle avec GPU.
- **Streamlit** : création de l'interface utilisateur.

## 📊 Jeu de données

Le modèle a été entraîné à partir du dataset Roboflow suivant :

[Traffic Sign Detection — version 11](https://universe.roboflow.com/university-km5u7/traffic-sign-detection-yolov8-awuus/dataset/11)

Le dataset comprend **2 647 images** :

- Entraînement : 2 352 images.
- Validation : 219 images.
- Test : 76 images.

Le modèle reconnaît 10 catégories de panneaux :

1. Dos-d'âne (`hump`)
2. Sens interdit (`no entry`)
3. Interdiction de dépasser (`no overtaking`)
4. Arrêt interdit (`no stopping`)
5. Demi-tour interdit (`no u turn`)
6. Stationnement (`parking`)
7. Travaux (`roadwork`)
8. Rond-point (`roundabout`)
9. Limitation de vitesse à 40 (`speed limit 40`)
10. Stop (`stop`)

## 📈 Résultats de l'évaluation

Les résultats obtenus sur le jeu de test sont les suivants :

| Métrique | Résultat |
|---|---:|
| Précision (Precision) | 90,33 % |
| Rappel (Recall) | 92,81 % |
| mAP@50 | 94,16 % |
| mAP@50-95 | 84,43 % |

### Interprétation

Le modèle obtient un mAP@50 de 94,16 % sur le jeu de test utilisé. Ces résultats sont encourageants pour un prototype de détection de panneaux routiers.

**Limite importante :** le jeu de test ne contient que 76 images et 79 objets annotés. Les performances doivent donc être interprétées avec prudence. Une évaluation sur un jeu de données plus vaste et plus diversifié permettrait de mieux mesurer la capacité de généralisation du modèle.

## 📁 Structure du projet

```text
Assistant_routier/
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── requirements.txt
├── README.md
├── notebooks/
│   └── train_roadsign.ipynb
└── models/
    └── best.pt  (modèle local, non envoyé sur GitHub)
```

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Eltaz100/Assistant_routier.git
cd Assistant_routier
```

### 2. Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

### 3. Ajouter le modèle entraîné

Placer le fichier `best.pt` dans le dossier `models/`.

Le fichier des poids du modèle n'est pas inclus dans le dépôt GitHub. Il doit être conservé séparément.

## ▶️ Utilisation

### Effectuer une prédiction

```bash
python predict.py --source chemin/vers/image.jpg
```

### Évaluer le modèle

```bash
python evaluate.py
```

### Lancer l'interface

```bash
python -m streamlit run app.py
```

Les commandes devront être vérifiées par rapport aux arguments réellement acceptés par les scripts du projet avant la publication finale.

## 🔮 Améliorations futures

- Tester le modèle sur davantage d'images réelles.
- Améliorer les performances sur les classes moins bien reconnues.
- Développer une interface interactive avec Streamlit.
- Améliorer la présentation des résultats de détection.
- Étudier la possibilité d'une utilisation en temps réel.

## ⚠️ Limites du projet

Ce projet est un prototype à vocation pédagogique et démonstrative. Il ne constitue pas un système certifié de sécurité routière et ne doit pas être utilisé comme unique référence pour prendre des décisions de conduite.

## 👨‍💻 Auteur

Projet personnel de vision par ordinateur consacré à la détection de panneaux routiers avec Python et YOLO.
