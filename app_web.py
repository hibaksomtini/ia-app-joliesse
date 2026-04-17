import streamlit as st
import pg8000
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import requests
from io import BytesIO
from streamlit_cropper import st_cropper

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Joliesse IA - Dashboard", layout="wide")

# Paramètres de connexion (Identiques à votre app Desktop)
DB_CONFIG = {
    "user": "postgres.mcmwrchllpqokgcdzmhl",
    "password": "Joliesse@123456",
    "host": "aws-0-eu-west-1.pooler.supabase.com",
    "database": "postgres",
    "port": 6543
}
STORAGE_URL = "https://mcmwrchllpqokgcdzmhl.supabase.co/storage/v1/object/public/catalogue/"

# --- CHARGEMENT DU MODÈLE IA ---
@st.cache_resource
def load_model():
    return SentenceTransformer('clip-ViT-L-14')

model = load_model()

# --- LOGIQUE DE RECHERCHE ET RÉPARATION ---
def run_search(processed_image):
    # Encodage de l'image cropée
    embedding = model.encode(processed_image, convert_to_numpy=True)
    norm = np.linalg.norm(embedding)
    query_vec = (embedding / norm).flatten()

    try:
        conn = pg8000.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # On récupère tout pour comparer
        cur.execute("SELECT product_ref, price, image_paths, embedding, colors FROM products")
        rows = cur.fetchall()

        results = []
        for ref, price, img_data, db_emb,colors in rows:
            # RÉPARATION AUTOMATIQUE (Votre logique ✅)
            if db_emb is None or np.array(db_emb).shape[0] != 768:
                print(f"🛠️ Réparation du vecteur pour : {ref}")
                try:
                    img_name = str(img_data).split('|')[0].strip()
                    resp = requests.get(STORAGE_URL + img_name, timeout=5)
                    if resp.status_code == 200:
                        temp_img = Image.open(BytesIO(resp.content))
                        new_emb = model.encode(temp_img).tolist()
                        
                        # Formatage compatible pgvector
                        formatted_vector = "[" + ",".join(map(str, new_emb)) + "]"
                        cur.execute("UPDATE products SET embedding = %s WHERE product_ref = %s", (formatted_vector, ref))
                        conn.commit()
                        db_emb = new_emb
                        st.toast(f"✅ Vecteur réparé : {ref}")
                except Exception:
                    conn.rollback() # Débloque la transaction en cas d'erreur
                    continue

            # CALCUL DE SIMILARITÉ
            if db_emb:
                db_vec = np.array(db_emb).flatten()
                # Si le vecteur est rempli de zéros ou vide, on l'ignore
                if db_vec.shape[0] != 768 or np.all(db_vec == 0):
                    continue
                score = np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec))
                
                img_list = str(img_data).split('|') if img_data else []
                if len(img_list) > 0:
                    first_image = img_list[0].strip()
                else:
                    first_image = "default.jpg" # Ou ignorer le produit
                results.append({
                    "ref": ref, 
                    "score": float(score), 
                    "price": price, 
                    "images": img_list
                })

        conn.close()
        # Tri par meilleur score
        return sorted(results, key=lambda x: x["score"], reverse=True)[:10]
    except Exception as e:
        st.error(f"Erreur de connexion base de données : {e}")
        return []

# --- INTERFACE UTILISATEUR ---
st.title("👟 JOLIESSE IA - Système de Détection")

# Menu de navigation
menu = st.sidebar.radio("Menu", ["🔍 Recherche", "📦 Catalogue"])

if menu == "🔍 Recherche":
    uploaded_file = st.file_uploader("Charger une photo de chaussure", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
    
        # Création de deux onglets
        tab_direct, tab_crop = st.tabs(["🚀 Scan Direct", "✂️ Recadrage Manuel"])

        with tab_direct:
            st.info("L'IA analyse l'image entière. Idéal pour une photo centrée.")
            st.image(img, use_container_width=True)
            if st.button("LANCER LE SCAN DIRECT", type="primary", key="btn_direct"):
                with st.spinner("Analyse globale..."):
                    st.session_state['results'] = run_search(img)

        with tab_crop:
            st.info("Ajustez le cadre rouge sur la chaussure précise.")
            # Utilisation de st_cropper
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000', aspect_ratio=None, key="cropper_manual")
            if st.button("SCANNER LA SÉLECTION", type="primary", key="btn_crop"):
                with st.spinner("Analyse de la zone..."):
                    st.session_state['results'] = run_search(cropped_img)

        # Affichage des résultats en dessous des onglets
        if 'results' in st.session_state:
            st.divider()
            st.subheader("Résultats de l'analyse")
            # ... (votre code d'affichage des colonnes reste le même)

elif menu == "📦 Catalogue":
    st.subheader("Explorateur de stock")
    search_ref = st.text_input("🔍 Rechercher par référence", placeholder="Ex: 4414")
    
    try:
        conn = pg8000.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        if search_ref:
            cur.execute("SELECT product_ref, price, image_paths, embedding, colors FROM products WHERE product_ref ILIKE %s LIMIT 50", (f"%{search_ref}%",))
        else:
            cur.execute("SELECT product_ref, price, image_paths, embedding, colors FROM products ORDER BY product_ref ASC LIMIT 50")
        
        rows = cur.fetchall()
        conn.close()

        for ref, price, img_data, db_emb, colors in rows:
            with st.expander(f"Référence : {ref}"):
                c1, c2 = st.columns([1, 3])
                if img_data:
                    c1.image(STORAGE_URL + str(img_data).split('|')[0].strip(), width=150)
                p_cat = f"{price:.2f} DT" if price is not None else "Non renseigné"
                c2.write(f"**Prix actuel :** {p_cat}")
                c2.button(f"Éditer {ref}", key=f"btn_{ref}")
                
    except Exception as e:
        st.error(f"Impossible de charger le catalogue : {e}")