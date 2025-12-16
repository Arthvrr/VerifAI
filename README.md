# 🛡️ VerifAI

**VerifAI** est un outil expérimental codé en Python conçu pour différencier une image réelle (photographie) d'une image générée par une Intelligence Artificielle (comme Midjourney, Flux ou DALL-E).

Contrairement aux "boîtes noires" basées sur le Deep Learning complexe, VerifAI utilise une approche transparente basée sur la **Vision par Ordinateur (Computer Vision)** et l'analyse statistique.

---

## 🧠 Comment ça marche ?

Les modèles de génération d'images laissent souvent des traces subtiles dans la structure des pixels : des gradients trop parfaits, un bruit numérique spécifique ou des artefacts de haute fréquence.

**L'algorithme de VerifAI suit ces étapes :**

1.  **Extraction des Gradients (Filtres de Sobel) :** L'outil analyse les variations d'intensité lumineuse (les contours) de l'image pour comprendre sa texture.
2.  **Réduction de dimension (PCA) :** Les données complexes de l'image sont compressées via une *Analyse en Composantes Principales* pour ne garder que les caractéristiques essentielles.
3.  **Comparaison Géométrique :** L'image testée est placée dans un espace vectoriel. L'algorithme mesure sa distance euclidienne par rapport au "centre de gravité" des images réelles et des images IA apprises.

---

## 📂 Structure du Projet

```text
VerifAI/
│
├── dataset/             # Généré automatiquement
│   ├── real/            # Photos réelles (via Picsum)
│   ├── ai/              # Images synthétiques (via Pollinations.ai)
│   └── tests/           # Placez ici vos images à tester !
│
├── get_images.py        # Script de constitution du Dataset
├── main.py              # Cœur du programme (Entraînement + Détection)
├── requirements.txt     # Liste des dépendances
└── README.md            # Documentation