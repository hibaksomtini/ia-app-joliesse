import streamlit as st
import pg8000
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import requests
from io import BytesIO
from streamlit_cropper import st_cropper
import warnings
from supabase import create_client, Client
import io
import time
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Joliesse IA - Dashboard", layout="wide")

# Paramètres de connexion
DB_CONFIG = {
    "user": "postgres.mcmwrchllpqokgcdzmhl",
    "password": "Joliesse@123456",
    "host": "aws-0-eu-west-1.pooler.supabase.com",
    "database": "postgres",
    "port": 6543

}
STORAGE_URL = "https://mcmwrchllpqokgcdzmhl.supabase.co/storage/v1/object/public/catalogue/"

SUPABASE_URL = "https://mcmwrchllpqokgcdzmhl.supabase.co"
SUPABASE_KEY = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- CHARGEMENT DU MODÈLE IA ---
@st.cache_resource
def load_model():
    return SentenceTransformer('clip-ViT-L-14')

model = load_model()

def get_cegid_data_and_colors(product_ref):
    """
    Récupère les déclinaisons Cegid, extrait la liste des couleurs uniques et le prix max/par défaut.
    """
    try:
        conn = pg8000.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        query = """
            SELECT barcode, color, size_label, price 
            FROM cegid_stocks 
            WHERE product_ref = %s 
            ORDER BY size_label ASC
        """
        cur.execute(query, (str(product_ref).strip(),))
        rows = cur.fetchall()
        conn.close()

        if rows:
            # 1. Extraction des couleurs uniques (en ignorant les N/A ou vides)
            unique_colors = sorted(list(set(
                str(row[1]).strip() for row in rows if row[1] and str(row[1]).strip() not in ["N/A", "nan", ""]
            )))
            
            # 2. Extraction du premier prix valide trouvé dans les déclinaisons Cegid
            cegid_prices = [float(row[3]) for row in rows if row[3] and float(row[3]) > 0]
            first_valid_price = cegid_prices[0] if cegid_prices else None
            
            # 3. Préparation du DataFrame pour le tableau
            df_variants = pd.DataFrame(rows, columns=['Code-barres', 'Couleur', 'Pointure/Taille', 'Prix (DT)'])
            df_variants['Prix (DT)'] = df_variants['Prix (DT)'].apply(lambda x: f"{float(x):.2f} DT" if x and str(x) != '-' else "-")
            
            return df_variants, unique_colors, first_valid_price
        return None, [], None
            
    except Exception as e:
        st.error(f"Erreur Cegid : {e}")
        return None, [], None

# --- LOGIQUE DE RECHERCHE ET RÉPARATION ---
def run_search(processed_image):
    embedding = model.encode(processed_image, convert_to_numpy=True)
    norm = np.linalg.norm(embedding)
    query_vec = (embedding / norm).flatten()

    try:
        conn = pg8000.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("SELECT product_ref, price, image_paths, embedding, colors FROM products")
        rows = cur.fetchall()
        conn.close()

        results = []
        st.write(f"Vérification de {len(rows)} produits...")
        for row in rows:
            if len(row) < 5:
                continue
                
            ref, price, img_data, db_emb, colors = row

            if db_emb is None:
                continue 

            try:
                if isinstance(db_emb, str):
                    import json
                    db_emb = json.loads(db_emb.replace("'", '"'))

                db_vec = np.array(db_emb).flatten()
                
                if db_vec.shape[0] != 768 or np.all(db_vec == 0):
                    continue

                score = np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec))

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
                continue

        return sorted(results, key=lambda x: x["score"], reverse=True)[:10]

    except Exception as e:
        st.error(f"Erreur de connexion base de données : {e}")
        return []

# --- INTERFACE UTILISATEUR ---
st.title("👟 JOLIESSE IA - Système de Détection")

# Menu de navigation
menu = st.sidebar.radio("Menu", ["🔍 Recherche", "📦 Catalogue", "🔐 Administration"])

if menu == "🔍 Recherche":
    uploaded_file = st.file_uploader("Charger une photo de chaussure", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
            st.session_state["last_uploaded_file"] = uploaded_file.name
            if 'results' in st.session_state:
                del st.session_state['results']
        
        img = Image.open(uploaded_file)
        mode = st.radio("Méthode d'analyse :", ["🚀 Scan Direct (Image entière)", "✂️ Recadrage Précis"], horizontal=True)

        if mode == "🚀 Scan Direct (Image entière)":
            st.info("L'IA analyse l'image entière.")
            st.image(img, use_container_width=True)
            if st.button("LANCER LE SCAN DIRECT", type="primary", key="btn_direct"):
                with st.spinner("Analyse globale..."):
                    st.session_state['results'] = run_search(img)

        else:
            st.warning("💡 Touchez l'image ci-dessous pour activer le cadre rouge.")
            img_display = img.copy().convert("RGB")
            img_display.thumbnail((1000, 1000)) 

            try:
                cropped_img = st_cropper(
                    img_display,
                    realtime_update=True,
                    box_color='#FF0000',
                    aspect_ratio=None,
                    key=f"cropper_stable_{uploaded_file.name}"
                )
                
                if cropped_img:
                    st.divider()
                    col_pre, col_act = st.columns([1, 1])
                    with col_pre:
                        st.write("🔍 **Aperçu du recadrage :**")
                        st.image(cropped_img, width=200)
                    with col_act:
                        st.write("⚡ **Action :**")
                        if st.button("LANCER L'ANALYSE DE LA ZONE", type="primary", key="btn_crop_final"):
                            with st.spinner("Recherche Joliesse..."):
                                st.session_state['results'] = run_search(cropped_img)
            
            except Exception as e:
                st.error(f"Le widget de recadrage ne répond pas : {e}")

        # Affichage des résultats (Recherche)
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

                        df_cegid, cegid_colors, first_cegid_price = get_cegid_data_and_colors(res['ref'])
                        
                        # Priorité de la couleur
                        if res.get('colors') and res['colors'] != "Non spécifié":
                            couleur_a_afficher = res['colors']
                        elif cegid_colors:
                            couleur_a_afficher = ", ".join(cegid_colors)
                        else:
                            couleur_a_afficher = "Non spécifiée"

                        if couleur_a_afficher != "Non spécifiée":
                            st.write(f"🎨 **Couleur(s) dispo :** {couleur_a_afficher}")
                        else:
                            st.caption("Couleur non spécifiée")

                        # Priorité du Prix (Recherche)
                        current_price = res.get('price')
                        if current_price is not None and float(current_price) > 0:
                            prix_a_afficher = float(current_price)
                        elif first_cegid_price:
                            prix_a_afficher = first_cegid_price
                        else:
                            prix_a_afficher = None

                        if prix_a_afficher:
                            st.write(f"Prix : :green[{prix_a_afficher:.2f} DT]")
                        else:
                            st.write("Prix : :orange[Non défini]")

                        score_val = res.get('score', 0.0)
                        if np.isnan(score_val):
                            score_val = 0.0

                        st.caption(f"Match : {score_val*100:.1f}%")
                        st.progress(min(max(float(score_val), 0.0), 1.0))
                        
                        st.divider()
                        if df_cegid is not None:
                            with st.expander(f"📊 Voir les déclinaisons & codes-barres Cegid ({len(df_cegid)})"):
                                st.dataframe(df_cegid, use_container_width=True, hide_index=True)
                        else:
                            st.caption("ℹ️ Aucune déclinaison Cegid trouvée pour cette référence.")

        with st.expander("📢 Un problème ? Article non trouvé ou erreur ?"):
            st.write("Aidez-nous à améliorer le catalogue Joliesse.")
            with st.form("feedback_form", clear_on_submit=True):
                type_msg = st.selectbox("Type de message :", ["🔍 Article non trouvé (Lancer l'ajout)", "⚠️ Erreur d'information (Prix/Couleur)", "💡 Suggestion d'amélioration"])
                user_comment = st.text_area("Détails (référence manquante, erreur constatée...)")
                feedback_img = st.file_uploader("Joindre une photo (si besoin)", type=['jpg', 'png'])

                if st.form_submit_button("ENVOYER LE SIGNALEMENT"):
                    if user_comment:
                        try:
                            file_name = None
                            if feedback_img:
                                import time
                                file_ext = feedback_img.name.split('.')[-1]
                                file_name = f"fb_{int(time.time())}.{file_ext}"
                                with st.spinner("Envoi de l'image..."):
                                    feedback_img.seek(0)
                                    supabase.storage.from_("catalogue").upload(path=f"feedbacks/{file_name}", file=feedback_img.read(), file_options={"content-type": f"image/{file_ext}"})

                            conn = pg8000.connect(**DB_CONFIG)
                            cur = conn.cursor()
                            sql = "INSERT INTO feedbacks (type_message, commentaire, status, attachment_path) VALUES (%s, %s, %s, %s)"
                            cur.execute(sql, (type_msg, user_comment, "Nouveau", file_name))
                            conn.commit()
                            conn.close()
                            st.success("✅ Signalement envoyé avec succès !")
                        except Exception as e:
                            st.error(f"Erreur lors de l'envoi : {e}")  

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
                ref = row[0]
                price = row[1]
                img_data = row[2]
                db_emb = row[3]
                colors = row[4]
                
                with st.expander(f"📦 Référence : {ref}"):
                    c1, c2 = st.columns([1, 2])
            
                    with c1:
                        img_list = str(img_data).split('|') if img_data else []
                        if img_list:
                            full_url = STORAGE_URL + img_list[0].strip()
                            st.image(full_url, width='stretch')
                        else:
                            st.info("Aucune image")

                    with c2:
                        # =============================================================
                        # 🌟 CHARGEMENT DYNAMIQUE ET FUSION DE COULEURS & PRIX (CEGID)
                        # =============================================================
                        df_cegid, cegid_colors, first_cegid_price = get_cegid_data_and_colors(ref)
                        
                        # Détermination du prix final (Priorité Base principale -> Cegid)
                        if price is not None and float(price) > 0:
                            final_catalog_price = float(price)
                        elif first_cegid_price:
                            final_catalog_price = first_cegid_price
                        else:
                            final_catalog_price = None

                        # Affichage du prix synchronisé
                        if final_catalog_price:
                            st.write(f"**Prix :** :green[{final_catalog_price:.2f} DT]")
                        else:
                            st.write("**Prix :** :orange[Non défini]")
                        
                        # Détermination de la couleur
                        if colors and colors != "None" and colors != "Non spécifié":
                            couleur_catalogue = colors
                        elif cegid_colors:
                            couleur_catalogue = ", ".join(cegid_colors)
                        else:
                            couleur_catalogue = "Non spécifiée"

                        if couleur_catalogue != "Non spécifiée":
                            st.write(f"🎨 **Couleur(s) dispo :** {couleur_catalogue}")
                        else:
                            st.caption("🎨 Couleur : Non spécifiée")
                        # =============================================================
                
                        if db_emb is not None:
                            st.success("✅ Prêt pour l'analyse IA")
                        else:
                            st.warning("⚠️ Non indexé (IA inactive)")
                            if st.button(f"Générer l'index pour {ref}", key=ref):
                                st.info("Traitement en cours...")

                        st.divider()
                        if df_cegid is not None:
                            with st.container():
                                st.write("📊 **Déclinaisons & codes-barres Cegid :**")
                                st.dataframe(df_cegid, use_container_width=True, hide_index=True)
                        else:
                            st.caption("ℹ️ Aucune déclinaison Cegid trouvée pour cette référence.")

            except IndexError as e:
                st.error(f"💥 INDEX ERROR : Manque de colonnes dans la table SQL ({len(row)} trouvées).")
                break
                
    except Exception as e:
        st.error(f"Impossible de charger le catalogue : {e}")

elif menu == "🔐 Administration":
    if 'upload_history' not in st.session_state:
        st.session_state['upload_history'] = []

    st.subheader("Gestion du Catalogue & Indexation IA")
    
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False

    if not st.session_state["admin_authenticated"]:
        pwd = st.text_input("Code d'accès sécurisé", type="password")
        if st.button("Se connecter"):
            if pwd == "Joliesse@2026":
                st.session_state["admin_authenticated"] = True
                st.rerun()
            else:
                st.error("Accès refusé.")
    else:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.success("Mode Administrateur Activé")
        with col2:
            if st.button("Se déconnecter"):
                st.session_state["admin_authenticated"] = False
                st.rerun()

        st.divider()
        st.write("### 📊 Importation et Synchronisation Cegid (Optimisée)")
        st.info("Glissez vos fichiers Excel. Le traitement est désormais optimisé en bloc.")

        cegid_files = st.file_uploader("Choisir les fichiers Excel Cegid", type=['xlsx'], accept_multiple_files=True, key="cegid_multi_uploader")

        if cegid_files:
            if st.button("🚀 LANCER LA SYNCHRONISATION CEGID", type="primary"):
                try:
                    conn = pg8000.connect(**DB_CONFIG)
                    cur = conn.cursor()
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS cegid_stocks (
                            id SERIAL PRIMARY KEY,
                            product_ref VARCHAR(50) NOT NULL,
                            barcode VARCHAR(50) UNIQUE NOT NULL,
                            color VARCHAR(50),
                            size_label VARCHAR(50),
                            price NUMERIC(10, 2),
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    conn.commit()

                    all_rows_to_upsert = []
                    for uploaded_file in cegid_files:
                        with st.spinner(f"Préparation de : {uploaded_file.name}..."):
                            df = pd.read_excel(uploaded_file, dtype={'Code article': str, 'Code-barres article': str, 'Couleur': str, 'Libellé dimension': str})
                            current_parent_price = 0.0

                            for index, row in df.iterrows():
                                ref = str(row.get('Code article', '')).strip()
                                barcode = str(row.get('Code-barres article', '')).strip()
                                color = str(row.get('Couleur', '')).strip()
                                size = str(row.get('Libellé dimension', '')).strip()
                                price_val = row.get('Prix Détail (TTC)')

                                if (barcode == '' or barcode == 'nan') and ref != '' and ref != 'nan':
                                    if pd.notna(price_val):
                                        current_parent_price = float(price_val)
                                    continue

                                if barcode == '' or barcode == 'nan' or ref == '' or ref == 'nan':
                                    continue

                                if color == 'nan' or color == '': color = "N/A"
                                if size == 'nan' or size == '': size = "N/A"
                                
                                final_price = float(price_val) if pd.notna(price_val) else current_parent_price
                                all_rows_to_upsert.append((ref, barcode, color, size, final_price))

                    if all_rows_to_upsert:
                        with st.spinner(f"Envoi de {len(all_rows_to_upsert)} déclinaisons à la base..."):
                            sql_upsert = """
                                INSERT INTO cegid_stocks (product_ref, barcode, color, size_label, price)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (barcode) 
                                DO UPDATE SET 
                                    product_ref = EXCLUDED.product_ref,
                                    color = EXCLUDED.color,
                                    size_label = EXCLUDED.size_label,
                                    price = EXCLUDED.price,
                                    updated_at = CURRENT_TIMESTAMP;
                            """
                            cur.executemany(sql_upsert, all_rows_to_upsert)
                            conn.commit()
                            st.success(f"✅ Synchronisation réussie ! {len(all_rows_to_upsert)} lignes traitées.")
                    else:
                        st.warning("⚠️ Aucun code-barres valide trouvé.")

                    conn.close()
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors de l'intégration : {e}")

        st.divider()
        st.write("### 📩 Messages et Alertes")
        try:
            conn = pg8000.connect(**DB_CONFIG)
            df_fb = pd.read_sql("SELECT id, created_at, type_message, commentaire, status, attachment_path FROM feedbacks WHERE status = 'Nouveau' ORDER BY id DESC", conn)
            if not df_fb.empty:
                st.dataframe(df_fb, use_container_width=True, hide_index=True)
                if st.button("Marquer ces messages comme 'Lus'"):
                    cur = conn.cursor()
                    cur.execute("UPDATE feedbacks SET status = 'Lu' WHERE status = 'Nouveau'")
                    conn.commit()
                    st.success("Messages archivés !")
                    time.sleep(1)
                    st.rerun()
            else:
                st.info("Aucun nouveau message.")
            conn.close()
        except Exception as e:
            st.info("La table des messages est en attente.")

        st.divider()
        st.write("### 🗃️ Gestion du Stock (Import Unique)")
        with st.form("admin_smart_upload", clear_on_submit=True):
            new_ref = st.text_input("Référence de l'article (ex: 4420)")
            new_file = st.file_uploader("Ajouter une image", type=['jpg', 'jpeg', 'png'])
            submitted = st.form_submit_button("LANCER L'INDEXATION")

        if submitted:
            if new_ref and new_file:
                try:
                    image = Image.open(new_file).convert("RGB")
                    with st.spinner("Analyse CLIP..."):
                        embedding = model.encode(image).tolist()
                    
                    file_ext = new_file.name.split('.')[-1]
                    unique_name = f"{new_ref}_{int(time.time())}.{file_ext}"

                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=85)
                    buffer_data = buffer.getvalue()

                    conn = pg8000.connect(**DB_CONFIG)
                    cur = conn.cursor()
                    cur.execute("SELECT image_paths FROM products WHERE product_ref = %s", (new_ref,))
                    row = cur.fetchone()

                    action_type = ""
                    if row:
                        current_paths = row[0] if row[0] else ""
                        updated_paths = f"{current_paths}|{unique_name}" if current_paths else unique_name
                        sql = "UPDATE products SET image_paths = %s, embedding = %s WHERE product_ref = %s"
                        cur.execute(sql, (updated_paths, str(embedding), new_ref))
                        action_type = "Mise à jour (Image ajoutée)"
                    else:
                        sql = "INSERT INTO products (product_ref, image_paths, embedding) VALUES (%s, %s, %s)"
                        cur.execute(sql, (new_ref, unique_name, str(embedding)))
                        action_type = "Nouveau Produit Créé"

                    supabase.storage.from_("catalogue").upload(path=unique_name, file=buffer_data, file_options={"content-type": "image/jpeg"})
                    conn.commit()
                    conn.close()

                    new_entry = {'Heure': time.strftime("%H:%M:%S"), 'Référence': new_ref, 'Action': action_type, 'Fichier': unique_name}
                    st.session_state['upload_history'].insert(0, new_entry)
                    st.session_state['upload_history'] = st.session_state['upload_history'][:10]
                    st.success(f"✅ Article {new_ref} traité avec succès !")
                except Exception as e:
                    st.error(f"Erreur technique : {e}")
            else:
                st.warning("Veuillez remplir la référence ET choisir une image.")

        if st.session_state['upload_history']:
            st.divider()
            st.subheader("⏱️ Historique Récent")
            last = st.session_state['upload_history'][0]
            if "Nouveau" in last['Action']:
                st.success(f"🆕 **Nouveau produit créé** : Réf {last['Référence']}")
            else:
                st.info(f"🔄 **Image ajoutée** à la référence : {last['Référence']}")

            df_hist = pd.DataFrame(st.session_state['upload_history'])
            st.dataframe(df_hist, use_container_width=True, hide_index=True)