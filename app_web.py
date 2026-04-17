import streamlit as st
import pg8000
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import requests
from io import BytesIO
from streamlit_cropper import st_cropper
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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
        if rows:
            # LOG DE DEBUG : Affiche combien d'éléments il y a dans une ligne
            st.write(f"Contenu échantillon : {rows[0]}")
        results = []
        for row in rows:
            # Sécurité 1 : Vérifier qu'on a bien 5 éléments avant de déballer
            if len(row) < 5:
                continue
                
            ref, price, img_data, db_emb, colors = row

            # Sécurité 2 : Gérer le cas où l'embedding est absent (None)
            if db_emb is None:
                # Optionnel : lancer la réparation ici si vous voulez
                continue 

            db_vec = np.array(db_emb).flatten()
            
            # Sécurité 3 : Vérifier la taille du vecteur (768 pour CLIP)
            if db_vec.shape[0] != 768:
                continue

            # Calcul du score seulement si tout est OK
            try:
                score = np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec))
                
                img_list = str(img_data).split('|') if img_data else []
                
                results.append({
                    "ref": ref, 
                    "score": float(score), 
                    "price": price if price is not None else 0.0, 
                    "images": img_list,
                    "colors": colors if colors is not None else "N/A" # On ajoute colors ici
                })
            except Exception:
                continue

            # CALCUL DE SIMILARITÉ
            if db_emb:
                db_vec_check = np.array(db_emb) if db_emb is not None else np.array([])
                # Si le vecteur est rempli de zéros ou vide, on l'ignore
                if db_emb is None or db_vec_check.ndim == 0 or db_vec_check.shape[0] != 768:
                    continue
                score = np.dot(query_vec, db_vec_check) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec_check))
                
                img_list = str(img_data).split('|') if img_data else []
                if len(img_list) > 0:
                    first_image = img_list[0].strip()
                else:
                    first_image = "default.jpg" # Ou ignorer le produit
                results.append({
                    "ref": ref, 
                    "score": float(score), 
                    "price": price if price is not None else 0.0, # Sécurité pour le prix
                    "images": img_list if img_list else [],       # Sécurité pour les images
                    "colors": colors if colors is not None else "Non spécifié"
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
            st.image(img, width='stretch')
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
            for res in st.session_state['results']:
                with st.container(border=True):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        if res.get('images') and len(res['images']) > 0:
                            img_url = STORAGE_URL + res['images'][0].strip()
                            st.image(img_url, width='stretch')
                        else:
                            st.warning("Pas d'image")
                    with c2:
                        st.write(f"### REF: {res['ref']}")

                        if res['colors'] != "Non spécifié":
                            st.write(f"🎨 **Couleur :** {res['colors']}")
                        else:
                            st.caption("Couleur non spécifiée")

                        # Gestion du prix
                        if res['price'] > 0:
                            st.write(f"Prix : :green[{res['price']:.2f} DT]")
                        else:
                            st.write("Prix : :orange[Sur devis]")
                        score_val = res.get('score', 0.0)

                        if np.isnan(score_val):
                            score_val = 0.0

                        # Barre de progression pour le score de match
                        st.caption(f"Match : {score_val*100:.1f}%")
                        st.progress(min(max(float(score_val), 0.0), 1.0))

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
        for row in rows:
            try:
                # Tentative de lecture sécurisée
                ref = row[0]
                price = row[1]
                img_data = row[2]
                db_emb = row[3]
                colors = row[4]
                
                with st.expander(f"📦 Référence : {ref}"):
                    c1, c2 = st.columns([1, 2])
            
                    with c1:
                        # Affichage de l'image du catalogue
                        img_list = str(img_data).split('|') if img_data else []
                        if img_list:
                            full_url = STORAGE_URL + img_list[0].strip()
                            st.image(full_url, width='stretch')
                        else:
                            st.info("Aucune image")

                    with c2:
                        # Affichage des détails
                        st.write(f"**Prix :** {f'{price:.2f} DT' if price else ':orange[Non défini]'}")
                
                        # Affichage de la couleur (notre nouvelle colonne !)
                        if colors and colors != "None":
                            st.write(f"🎨 **Couleur :** {colors}")
                        else:
                            st.caption("🎨 Couleur : Non spécifiée")
                
                        # Statut de l'IA (Vecteur)
                        if db_emb is not None:
                            st.success("✅ Prêt pour l'analyse IA")
                        else:
                            st.warning("⚠️ Non indexé (IA inactive)")
                            if st.button(f"Générer l'index pour {ref}", key=ref):
                                st.info("Traitement en cours...")
            except IndexError as e:
                st.error(f"💥 INDEX ERROR : Vous essayez d'accéder à la colonne 5 (colors), mais la ligne n'a que {len(row)} colonnes.")
                st.write(f"Contenu de la ligne SQL : {row}")
                break # Arrête la boucle pour ne pas flooder l'écran d'erreurs
                
    except Exception as e:
        st.error(f"Impossible de charger le catalogue : {e}")