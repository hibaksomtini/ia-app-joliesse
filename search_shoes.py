import pg8000
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image

# 1. Image que vous voulez chercher (mettez le chemin vers une photo test)
image_a_chercher = r"C:\Users\lenovo\Desktop\test_image.JPEG"

try:
    # Connexion
    conn = pg8000.connect(user="postgres", password=r"JoliesseSql**Admin", host="localhost", database="postgres", port=5433)
    cur = conn.cursor()

    # 2. Transformation de l'image de test en vecteur
    ibed = imgbeddings()
    img = Image.open(image_a_chercher).convert('RGB')
    query_embedding = ibed.to_embeddings(img)[0].tolist()

    # 3. Récupération de tout le catalogue pour comparer
    cur.execute("SELECT product_ref, embedding FROM shoes")
    rows = cur.fetchall()

    resultats = []
    for ref, db_embedding in rows:
        # Calcul de la distance entre l'image test et l'image en base
        dist = np.linalg.norm(np.array(query_embedding) - np.array(db_embedding))
        resultats.append((ref, dist))

    # 4. Tri par ressemblance (la plus petite distance en premier)
    resultats.sort(key=lambda x: x[1])

    print(f"\n--- Résultats pour : {image_a_chercher} ---")
    for i, (ref, score) in enumerate(resultats[:5]): # On affiche les 5 meilleurs
        print(f"{i+1}. {ref} (Score de similarité : {score:.2f})")

    cur.close()
    conn.close()

except Exception as e:
    print(f"Erreur : {e}")