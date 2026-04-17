import pg8000
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

# --- CONFIGURATION ---
DB_CONFIG = {
    "user": "postgres",
    "password": r"JoliesseSql**Admin",
    "host": "localhost",
    "database": "postgres",
    "port": 5433
}
FOLDER_PATH = r"C:\Users\lenovo\Desktop\catalogue-IA"
EXCEL_PATH = os.path.join(FOLDER_PATH, "catalogue_data.xlsx")

def run_full_import():
    try:
        # 1. Connexion à la base de données
        conn = pg8000.connect(**DB_CONFIG)
        cur = conn.cursor()

        # 2. VIDER LA TABLE (On la recrée pour être sûr d'avoir les bonnes colonnes)
        print("🧹 Nettoyage de la base de données...")
        cur.execute("DROP TABLE IF EXISTS products CASCADE;")
        cur.execute("""
            CREATE TABLE products (
                product_ref TEXT PRIMARY KEY,
                price DECIMAL(10, 2) DEFAULT 0.0,
                colors TEXT[] DEFAULT '{}',
                image_paths TEXT[] DEFAULT '{}',
                embedding FLOAT8[]
            );
        """)
        conn.commit()

        # 3. Chargement de l'IA et de l'Excel
        print("🚀 Chargement du modèle IA...")
        model = SentenceTransformer('clip-ViT-L-14')
        df = pd.read_excel(EXCEL_PATH)

        print(f"🔎 {len(df)} produits trouvés dans l'Excel. Début du traitement...")

        for index, row in df.iterrows():
            ref = str(row['reference']).strip()
            
            # Découpage des images (séparées par une virgule dans votre Excel)
            image_list = [img.strip() for img in str(row['image_names']).split(',')] if pd.notna(row['image_names']) else []
            
            # Gestion des colonnes optionnelles (si elles existent dans l'Excel)
            price = float(row['price']) if 'price' in df.columns and pd.notna(row['price']) else 0.0
            colors = [c.strip() for c in str(row['colors']).split(',')] if 'colors' in df.columns and pd.notna(row['colors']) else []

            embedding = None
            if image_list:
                # On utilise la PREMIÈRE image pour créer le vecteur de recherche
                main_image_path = os.path.join(FOLDER_PATH, image_list[0])
                
                if os.path.exists(main_image_path):
                    try:
                        img_obj = Image.open(main_image_path).convert('RGB')
                        # Encodage et normalisation du vecteur
                        vector = model.encode(img_obj)
                        embedding = (vector / np.linalg.norm(vector)).tolist()
                    except Exception as e:
                        print(f"⚠️ Erreur sur l'image {image_list[0]} : {e}")
                else:
                    print(f"❓ Fichier introuvable : {image_list[0]}")

            # 4. INSERTION DANS LA TABLE
            cur.execute("""
                INSERT INTO products (product_ref, price, colors, image_paths, embedding)
                VALUES (%s, %s, %s, %s, %s)
            """, (ref, price, colors, image_list, embedding))

            if index % 20 == 0:
                conn.commit()
                print(f"📊 Progression : {index}/{len(df)} articles...")

        conn.commit()
        print("\n✨ IMPORTATION TERMINÉE AVEC SUCCÈS !")

    except Exception as e:
        print(f"💥 Erreur critique : {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    run_full_import()