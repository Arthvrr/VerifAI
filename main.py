import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class HybridAIDetector:
    def __init__(self, image_size=(128, 128)):
        # On réduit légèrement la taille (128x128) pour garder de la vitesse 
        # car la FFT + Sobel génère beaucoup de données.
        self.image_size = image_size
        self.pca = PCA(n_components=2)
        self.scaler = StandardScaler()
        self.train_data_pca = None
        self.train_labels = None
        self.is_trained = False

    def _get_features(self, image_path):
        """
        Extrait une signature hybride : 
        50% Analyse de contours (Sobel) + 50% Analyse Fréquentielle (FFT)
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError(f"Erreur lecture: {image_path}")
        img = cv2.resize(img, self.image_size)

        # --- 1. ANALYSE DE GRADIENT (SOBEL) ---
        # Détecte la netteté artificielle et les contours
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_features = np.sqrt(sobelx**2 + sobely**2).flatten()

        # --- 2. ANALYSE FRÉQUENTIELLE (FFT) ---
        # Détecte les patterns répétitifs invisibles (artefacts de grille)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        # Magnitude spectrum (échelle log pour voir les détails)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)
        fft_features = magnitude_spectrum.flatten()

        # --- FUSION DES DONNÉES ---
        # On colle les deux vecteurs l'un à la suite de l'autre
        return np.concatenate([gradient_features, fft_features])

    def train(self, real_folder, ai_folder):
        print("🛠️  Entraînement HYBRIDE (Sobel + FFT)...")
        data = []
        self.train_labels = [] # 0=Réel, 1=IA
        
        # Fonction interne pour charger un dossier
        def load_folder(folder, label, label_name):
            count = 0
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(('jpg', 'png', 'jpeg', 'webp')):
                        try:
                            feat = self._get_features(os.path.join(folder, f))
                            data.append(feat)
                            self.train_labels.append(label)
                            count += 1
                        except Exception as e:
                            pass
            print(f"   -> {count} images {label_name} chargées.")
            return count

        c_real = load_folder(real_folder, 0, "RÉELLES")
        c_ai = load_folder(ai_folder, 1, "IA")
        
        if len(data) == 0:
            print("❌ Erreur : Pas assez de données.")
            return

        # Normalisation et Réduction de dimension
        print("   -> Optimisation mathématique en cours...")
        X = np.array(data)
        # Le Scaler est CRUCIAL ici car les valeurs FFT et Sobel n'ont pas la même échelle
        X_scaled = self.scaler.fit_transform(X)
        self.train_data_pca = self.pca.fit_transform(X_scaled)
        self.is_trained = True
        print("✅ Entraînement terminé.\n")

    def predict(self, image_path, show_plot=False):
        if not self.is_trained: return "Non entraîné", 0, 0
        
        try:
            # 1. Extraction et Transformation
            feat = self._get_features(image_path)
            feat_scaled = self.scaler.transform([feat])
            point = self.pca.transform(feat_scaled)[0]
            
            # 2. Calcul des distances aux centres
            real_points = self.train_data_pca[np.array(self.train_labels) == 0]
            ai_points = self.train_data_pca[np.array(self.train_labels) == 1]
            
            center_real = np.mean(real_points, axis=0)
            center_ai = np.mean(ai_points, axis=0)
            
            dist_real = np.linalg.norm(point - center_real)
            dist_ai = np.linalg.norm(point - center_ai)
            
            # Logique de décision simple
            verdict = "IA" if dist_ai < dist_real else "RÉELLE"
            
            # 3. Visualisation
            if show_plot:
                plt.figure(figsize=(10, 6))
                plt.scatter(real_points[:, 0], real_points[:, 1], c='blue', label='Real (Train)', alpha=0.5)
                plt.scatter(ai_points[:, 0], ai_points[:, 1], c='red', label='IA (Train)', alpha=0.5)
                plt.scatter(point[0], point[1], c='lime', s=250, marker='*', edgecolors='black', label='IMAGE TEST')
                
                # Cercles de distance (visuel sympa)
                circle_real = plt.Circle(center_real, dist_real, color='blue', fill=False, linestyle='--', alpha=0.3)
                circle_ai = plt.Circle(center_ai, dist_ai, color='red', fill=False, linestyle='--', alpha=0.3)
                plt.gca().add_patch(circle_real)
                plt.gca().add_patch(circle_ai)

                plt.title(f"VerifAI Hybrid: {os.path.basename(image_path)} -> {verdict}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
            
            return verdict, dist_real, dist_ai
            
        except Exception as e:
            print(f"Erreur sur {image_path}: {e}")
            return "Erreur", 0, 0

# --- PARTIE PRINCIPALE ---
if __name__ == "__main__":
    # Instanciation du détecteur Hybride
    detector = HybridAIDetector()
    
    dossier_real = "dataset/real"
    dossier_ai = "dataset/ai"
    dossier_test = "dataset/tests"

    if os.path.exists(dossier_real) and os.path.exists(dossier_ai):
        detector.train(dossier_real, dossier_ai)
        
        if os.path.exists(dossier_test):
            print(f"🔍 ANALYSE HYBRIDE (SOBEL + FFT) : {dossier_test}")
            print("-" * 65)
            print(f"{'FICHIER':<30} | {'VERDICT':<10} | {'DISTANCES (R/IA)':<20}")
            print("-" * 65)

            fichiers_test = os.listdir(dossier_test)
            stats = {"IA": 0, "RÉELLE": 0}

            for f in fichiers_test:
                if f.lower().endswith(('jpg', 'png', 'jpeg', 'webp')):
                    path = os.path.join(dossier_test, f)
                    res, d_real, d_ai = detector.predict(path, show_plot=False)
                    
                    if res in stats: stats[res] += 1
                    icone = "🤖" if res == "IA" else "📸"
                    
                    # Code couleur console (juste pour faire joli)
                    # \033[91m = Rouge, \033[92m = Vert, \033[0m = Reset
                    color = "\033[91m" if res == "IA" else "\033[92m"
                    print(f"{f:<30} | {icone} {color}{res:<8}\033[0m | R:{d_real:.2f} / IA:{d_ai:.2f}")

            print("-" * 65)
            print(f"📊 BILAN : {stats['IA']} IA détectées / {stats['RÉELLE']} Photos réelles détectées.")
        else:
            print(f"⚠️ Créez un dossier '{dossier_test}' et mettez des images dedans !")
    else:
        print("⚠️ Lance d'abord 'python get_images.py' pour créer le dataset !")