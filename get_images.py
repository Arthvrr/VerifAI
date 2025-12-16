import os
import requests
import time
import random
import uuid # Pour générer du texte unique

# --- CONFIGURATION ---
NUM_IMAGES = 50
SAVE_DIR_REAL = "dataset/real"
SAVE_DIR_AI = "dataset/ai"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def download_image(url, folder, filename):
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        print(f"⏩ Déjà là : {filename}")
        return True

    try:
        # Timeout long pour éviter les erreurs rouges
        response = requests.get(url, headers=HEADERS, timeout=60)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"✅ Téléchargé : {filename}")
            return True
        else:
            print(f"⚠️ Status {response.status_code}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    return False

def generate_chaos_prompt():
    """Génère un prompt totalement unique pour empêcher le cache."""
    subjects = ["woman face", "man face", "cyberpunk city", "cat", "dog", "forest", "car", "robot", "flower", "mountain"]
    styles = ["photorealistic", "cinematic lighting", "8k", "highly detailed", "dramatic"]
    colors = ["red", "blue", "green", "purple", "golden", "dark"]
    
    # On mélange tout + un identifiant unique (UUID) à la fin
    # Le serveur ne pourra JAMAIS avoir ce prompt en cache.
    prompt = f"{random.choice(subjects)}, {random.choice(styles)}, {random.choice(colors)} tone, {str(uuid.uuid4())[:8]}"
    return prompt

def main():
    os.makedirs(SAVE_DIR_REAL, exist_ok=True)
    os.makedirs(SAVE_DIR_AI, exist_ok=True)

    print(f"🚀 Démarrage... On va forcer la variété !")

    # 1. REAL (Picsum)
    print("\n--- PHOTOS RÉELLES ---")
    for i in range(NUM_IMAGES):
        # On change le seed picsum à chaque boucle
        url = f"https://picsum.photos/seed/{random.randint(1, 999999)}/500/500"
        download_image(url, SAVE_DIR_REAL, f"real_{i}.jpg")

    # 2. IA (Pollinations avec Chaos)
    print("\n--- IMAGES IA ---")
    for i in range(NUM_IMAGES):
        prompt = generate_chaos_prompt()
        print(f"🎨 Génération {i+1}/{NUM_IMAGES} : '{prompt}'")
        
        # On encode l'URL correctement
        url = f"https://image.pollinations.ai/prompt/{prompt}?width=500&height=500&nologo=true&model=flux"
        
        success = download_image(url, SAVE_DIR_AI, f"ai_{i}.jpg")
        
        if not success:
            print("   -> 🔄 Retry...")
            time.sleep(5)
            download_image(url, SAVE_DIR_AI, f"ai_{i}.jpg")
            
        time.sleep(3) # Pause anti-ban

    print("\n✨ Terminé ! Vérifie le dossier dataset/ai, elles devraient toutes être différentes.")

if __name__ == "__main__":
    main()