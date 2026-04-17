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
        
        col_input, col_output = st.columns([1, 1])
        
        with col_input:
            st.subheader("1. Recadrage")
            # Remplace votre rectangle rouge manuel par un outil de crop interactif
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
            
            if st.button("LANCER LA RECHERCHE", type="primary"):
                with st.spinner("Analyse du catalogue Joliesse..."):
                    st.session_state['results'] = run_search(cropped_img)

        with col_output:
            st.subheader("2. Résultats")
            if 'results' in st.session_state:
                for res in st.session_state['results']:
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            if res['images']:
                                st.image(STORAGE_URL + res['images'][0].strip())
                        with c2:
                            st.write(f"**REF: {res['ref']}**")
                            # Gestion du prix None (votre correction ✅)
                            p_display = f"{res['price']:.2f} DT" if res['price'] is not None else "--- DT"
                            st.write(f"Prix: :green[{p_display}]")
                            st.caption(f"Score de match : {res['score']*100:.1f}%")

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

        for ref, price, img_data in rows:
            with st.expander(f"Référence : {ref}"):
                c1, c2 = st.columns([1, 3])
                if img_data:
                    c1.image(STORAGE_URL + str(img_data).split('|')[0].strip(), width=150)
                p_cat = f"{price:.2f} DT" if price is not None else "Non renseigné"
                c2.write(f"**Prix actuel :** {p_cat}")
                c2.button(f"Éditer {ref}", key=f"btn_{ref}")
                
    except Exception as e:
        st.error(f"Impossible de charger le catalogue : {e}")