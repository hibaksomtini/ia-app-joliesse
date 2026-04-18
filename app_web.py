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
    # Encodage de l'image
    embedding = model.encode(processed_image, convert_to_numpy=True)
    norm = np.linalg.norm(embedding)
    query_vec = (embedding / norm).flatten()

    try:
        conn = pg8000.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Récupération des 5 colonnes nécessaires
        cur.execute("SELECT product_ref, price, image_paths, embedding, colors FROM products")
        rows = cur.fetchall()
        conn.close()

        results = []
        st.write(f"Vérification de {len(rows)} produits...")
        for row in rows:
            # Sécurité 1 : Structure de la ligne
            if len(row) < 5:
                continue
                
            ref, price, img_data, db_emb, colors = row

            # Sécurité 2 : On ignore si l'embedding est absent
            if db_emb is None:
                print(f"Skipping {ref}: Embedding is None")
                continue 

            try:
                if isinstance(db_emb, str):
                    import json
                    db_emb = json.loads(db_emb.replace("'", '"'))

                db_vec = np.array(db_emb).flatten()
                
                # Sécurité 3 : Taille du vecteur CLIP (768)
                if db_vec.shape[0] != 768 or np.all(db_vec == 0):
                    print(f"Dimension mismatch for {ref}: {db_vec.shape[0]}")
                    continue

                # CALCUL DE SIMILARITÉ
                score = np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec))
                
                print(f"Ref: {ref} | Score: {score:.4f}")

                # Formatage des images
                if score > 0.1: 
                    img_list = str(img_data).split('|') if img_data else []
                    results.append({
                        "ref": ref, 
                        "score": float(score), 
                        "price": float(price) if price is not None else 0.0,
                        "images": img_list,
                        "colors": colors if colors else "N/A"
                    })
            except Exception as e:
                print(f"Error processing {ref}: {e}")
                continue

        # Tri par meilleur score et limitation aux 10 meilleurs
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
        # --- MÉCANISME DE RÉINITIALISATION ---
        # On vérifie si c'est une nouvelle image par rapport à la dernière traitée
        if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
            st.session_state["last_uploaded_file"] = uploaded_file.name
            # On efface les anciens résultats pour forcer une nouvelle analyse
            if 'results' in st.session_state:
                del st.session_state['results']
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
            # On crée une clé unique par image pour forcer le reset seulement quand on change de fichier
            # Mais on garde la même clé tant qu'on travaille sur la même image
            cropper_key = f"cropper_{uploaded_file.name}"

            # On prépare l'image
            img_for_crop = img.copy().convert("RGB")
            img_for_crop.thumbnail((600, 600)) 

            # APPEL DU CROPPER
            # realtime_update=True est important pour garder le cadre visible
            cropped_img = st_cropper(
                img_for_crop, 
                realtime_update=True, 
                box_color='#FF0000', 
                aspect_ratio=None, 
                key=cropper_key,
                should_resize_canvas=True
            )
            
            if cropped_img:
                st.write("✅ **Zone sélectionnée :**")
                # On réduit l'aperçu pour ne pas encombrer l'écran
                st.image(cropped_img, width=150)

                # Utilisation d'un bouton avec une clé stable
                if st.button("LANCER L'ANALYSE DE LA SÉLECTION", type="primary", key="run_crop_analysis"):
                    with st.spinner("Analyse en cours..."):
                        # On lance la recherche et on stocke dans le session_state
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
                        current_price = res.get('price')
            
                        if current_price is not None and float(current_price) > 0:
                            st.write(f"Prix : :green[{float(current_price):.2f} DT]")
                        else:
                            st.write("Prix : :orange[Non défini]")

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