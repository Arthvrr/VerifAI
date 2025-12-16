import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class GradientAIDetector:
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size
        self.pca = PCA(n_components=2)
        self.scaler = StandardScaler()
        self.train_data_pca = None
        self.train_labels = None
        self.is_trained = False

    def _get_gradient_features(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError(f"Erreur lecture: {image_path}")
        img = cv2.resize(img, self.image_size)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2).flatten()

    def train(self, real_folder, ai_folder):
        print("🛠️  Entraînement et mémorisation des points...")
        data = []
        self.train_labels = [] # 0=Réel, 1=IA
        
        # Chargement images Réelles
        count_real = 0
        if os.path.exists(real_folder):
            for f in os.listdir(real_folder):
                if f.lower().endswith(('jpg', 'png', 'jpeg', 'webp')):
                    try:
                        data.append(self._get_gradient_features(os.path.join(real_folder, f)))
                        self.train_labels.append(0) # Label Réel
                        count_real += 1
                    except: pass
        
        # Chargement images IA
        count_ai = 0
        if os.path.exists(ai_folder):
            for f in os.listdir(ai_folder):
                if f.lower().endswith(('jpg', 'png', 'jpeg', 'webp')):
                    try:
                        data.append(self._get_gradient_features(os.path.join(ai_folder, f)))
                        self.train_labels.append(1) # Label IA
                        count_ai += 1
                    except: pass
        
        if len(data) == 0:
            print("❌ Erreur : Aucune image trouvée pour l'entraînement.")
            return

        print(f"   -> {count_real} images réelles chargées.")
        print(f"   -> {count_ai} images IA chargées.")

        X = np.array(data)
        X_scaled = self.scaler.fit_transform(X)
        self.train_data_pca = self.pca.fit_transform(X_scaled)
        self.is_trained = True
        print("✅ Entraînement terminé.\n")

    def predict(self, image_path, show_plot=False):
        """
        Predit si l'image est IA ou Réelle.
        show_plot=False par défaut pour ne pas bloquer la boucle.
        """
        if not self.is_trained: return "Non entraîné"
        
        try:
            # 1. Traitement image test
            feat = self._get_gradient_features(image_path)
            feat_scaled = self.scaler.transform([feat])
            point = self.pca.transform(feat_scaled)[0]
            
            # 2. Calcul des distances moyennes
            real_points = self.train_data_pca[np.array(self.train_labels) == 0]
            ai_points = self.train_data_pca[np.array(self.train_labels) == 1]
            
            center_real = np.mean(real_points, axis=0)
            center_ai = np.mean(ai_points, axis=0)
            
            dist_real = np.linalg.norm(point - center_real)
            dist_ai = np.linalg.norm(point - center_ai)
            
            verdict = "IA" if dist_ai < dist_real else "RÉELLE"
            
            # 3. VISUALISATION GRAPHIQUE (Optionnelle)
            if show_plot:
                plt.figure(figsize=(10, 6))
                plt.scatter(real_points[:, 0], real_points[:, 1], c='blue', label='Real (Train)', alpha=0.6)
                plt.scatter(ai_points[:, 0], ai_points[:, 1], c='red', label='IA (Train)', alpha=0.6)
                plt.scatter(point[0], point[1], c='green', s=200, marker='*', label='IMAGE TEST')
                plt.title(f"Analyse: {os.path.basename(image_path)} -> {verdict}")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.show()
            
            return verdict, dist_real, dist_ai
            
        except Exception as e:
            return "Erreur", 0, 0

# --- PARTIE PRINCIPALE ---
if __name__ == "__main__":
    detector = GradientAIDetector()
    
    # Dossiers d'entraînement
    dossier_real = "dataset/real"
    dossier_ai = "dataset/ai"
    
    # Dossier de test
    dossier_test = "dataset/tests"

    # Vérification des dossiers
    if os.path.exists(dossier_real) and os.path.exists(dossier_ai):
        detector.train(dossier_real, dossier_ai)
        
        if os.path.exists(dossier_test):
            print(f"🔍 DÉBUT DE L'ANALYSE DU DOSSIER : {dossier_test}")
            print("-" * 60)
            print(f"{'FICHIER':<30} | {'VERDICT':<10} | {'SCORE (Distances)':<20}")
            print("-" * 60)

            fichiers_test = os.listdir(dossier_test)
            detectes_ia = 0
            detectes_real = 0

            for f in fichiers_test:
                if f.lower().endswith(('jpg', 'png', 'jpeg', 'webp')):
                    chemin_complet = os.path.join(dossier_test, f)
                    
                    # On lance la prédiction sans afficher le graphique (show_plot=False)
                    res, d_real, d_ai = detector.predict(chemin_complet, show_plot=False)
                    
                    # Petites icones pour la lisibilité
                    icone = "🤖" if res == "IA" else "📸"
                    if res == "IA": detectes_ia += 1
                    else: detectes_real += 1

                    # Affichage formaté
                    print(f"{f:<30} | {icone} {res}   | R:{d_real:.2f} / IA:{d_ai:.2f}")

            print("-" * 60)
            print(f"📊 BILAN : {detectes_ia} détectés comme IA, {detectes_real} détectés comme PHOTOS RÉELLES.")
        else:
            print(f"⚠️ Le dossier de test '{dossier_test}' n'existe pas.")
    else:
        print("⚠️ Erreur : Les dossiers 'dataset/real' et 'dataset/ai' sont requis pour l'entraînement.")