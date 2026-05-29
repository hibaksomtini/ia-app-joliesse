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
    Récupère les déclinaisons Cegid (avec dépôts et stocks), extrait la liste des couleurs uniques et le prix.
    Gère l'insensibilité à la casse (ex: e25mudm135 vs E25MUDM135).
    """
    try:
        conn = pg8000.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        query = """
            SELECT depot_stock, color, size_label, price, stock_qty 
            FROM cegid_stocks 
            WHERE UPPER(product_ref) = %s 
            ORDER BY depot_stock ASC, size_label ASC
        """
        cur.execute(query, (str(product_ref).strip().upper(),))
        rows = cur.fetchall()
        conn.close()

        if rows:
            # 1. Extraction des couleurs uniques (sans doublons)
            unique_colors = sorted(list(set(
                str(row[1]).strip() for row in rows if row[1] and str(row[1]).strip() not in ["N/A", "nan", ""]
            )))
            
            # 2. Extraction du premier prix valide trouvé
            cegid_prices = [float(row[3]) for row in rows if row[3] and float(row[3]) > 0]
            first_valid_price = cegid_prices[0] if cegid_prices else None
            
            # 3. Préparation du DataFrame pour l'affichage complet
            df_variants = pd.DataFrame(rows, columns=['Dépôt / Magasin', 'Couleur', 'Pointure', 'Prix (DT)', 'Stock Qty'])
            df_variants['Prix (DT)'] = df_variants['Prix (DT)'].apply(lambda x: f"{float(x):.2f} DT" if x and str(x) != '-' else "-")
            df_variants['Stock Qty'] = df_variants['Stock Qty'].apply(lambda x: int(float(x)) if pd.notna(x) else 0)
            
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
            if len(row) < 5: continue
            ref, price, img_data, db_emb, colors = row
            if db_emb is None: continue 

            try:
                if isinstance(db_emb, str):
                    import json
                    db_emb = json.loads(db_emb.replace("'", '"'))

                db_vec = np.array(db_emb).flatten()
                if db_vec.shape[0] != 768 or np.all(db_vec == 0): continue

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

menu = st.sidebar.radio("Menu", ["🔍 Recherche", "📦 Catalogue", "🔐 Administration"])

if menu == "🔍 Recherche":
    uploaded_file = st.file_uploader("Charger une photo de chaussure", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
            st.session_state["last_uploaded_file"] = uploaded_file.name
            if 'results' in st.session_state: del st.session_state['results']
        
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
                cropped_img = st_cropper(img_display, realtime_update=True, box_color='#FF0000', aspect_ratio=None, key=f"cropper_stable_{uploaded_file.name}")
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

        if 'results' in st.session_state:
            st.divider()
            st.subheader("Résultats de l'analyse")
            for res in st.session_state['results']:
                with st.container(border=True):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        if res.get('images') and len(res['images']) > 0:
                            st.image(STORAGE_URL + res['images'][0].strip(), width='stretch')
                        else:
                            st.warning("Pas d'image")
                    with c2:
                        st.write(f"### REF: {res['ref']}")
                        df_cegid, cegid_colors, first_cegid_price = get_cegid_data_and_colors(res['ref'])
                        
                        # Couleurs
                        couleur_a_afficher = res['colors'] if res.get('colors') and res['colors'] != "Non spécifié" else (", ".join(cegid_colors) if cegid_colors else "Non spécifiée")
                        st.write(f"🎨 **Couleur(s) dispo :** {couleur_a_afficher}")

                        # Prix
                        prix_a_afficher = float(res['price']) if res.get('price') and float(res['price']) > 0 else first_cegid_price
                        st.write(f"Prix : :green[{f'{prix_a_afficher:.2f} DT' if prix_a_afficher else 'Non défini'}]")

                        # Total Stock global pour information rapide
                        if df_cegid is not None:
                            total_stock = df_cegid['Stock Qty'].sum()
                            st.write(f"📦 **Stock total disponible :** {total_stock} paires")

                        score_val = res.get('score', 0.0)
                        st.caption(f"Match : {score_val*100:.1f}%")
                        st.progress(min(max(float(score_val), 0.0), 1.0))
                        
                        st.divider()
                        if df_cegid is not None:
                            with st.expander(f"📊 Voir la disponibilité par Dépôt / Magasin ({len(df_cegid)} lignes)"):
                                st.dataframe(df_variants=df_cegid, use_container_width=True, hide_index=True)
                        else:
                            st.caption("ℹ️ Aucune donnée de dépôt trouvée pour cette référence.")

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
            ref, price, img_data, db_emb, colors = row[0], row[1], row[2], row[3], row[4]
            with st.expander(f"📦 Référence : {ref}"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    img_list = str(img_data).split('|') if img_data else []
                    if img_list: st.image(STORAGE_URL + img_list[0].strip(), width='stretch')
                    else: st.info("Aucune image")

                with c2:
                    df_cegid, cegid_colors, first_cegid_price = get_cegid_data_and_colors(ref)
                    
                    # Prix synchronisé
                    final_catalog_price = float(price) if price and float(price) > 0 else first_cegid_price
                    st.write(f"**Prix :** :green[{f'{final_catalog_price:.2f} DT' if final_catalog_price else 'Non défini'}]")
                    
                    # Couleur synchronisée
                    couleur_catalogue = colors if colors and colors != "None" and colors != "Non spécifié" else (", ".join(cegid_colors) if cegid_colors else "Non spécifiée")
                    st.write(f"🎨 **Couleur(s) dispo :** {couleur_catalogue}")
            
                    if db_emb is not None: st.success("✅ Prêt pour l'analyse IA")
                    else: st.warning("⚠️ Non indexé (IA inactive)")

                    st.divider()
                    if df_cegid is not None:
                        st.write("📊 **Disponibilité des stocks par Dépôt :**")
                        st.dataframe(df_cegid, use_container_width=True, hide_index=True)
                    else:
                        st.caption("ℹ️ Aucun stock dépôt enregistré.")
    except Exception as e:
        st.error(f"Impossible de charger le catalogue : {e}")

elif menu == "🔐 Administration":
    if 'upload_history' not in st.session_state: st.session_state['upload_history'] = []
    st.subheader("Gestion du Catalogue & Indexation IA")
    
    if "admin_authenticated" not in st.session_state: st.session_state["admin_authenticated"] = False

    if not st.session_state["admin_authenticated"]:
        pwd = st.text_input("Code d'accès sécurisé", type="password")
        if st.button("Se connecter"):
            if pwd == "Joliesse@2026":
                st.session_state["admin_authenticated"] = True
                st.rerun()
            else: st.error("Accès refusé.")
    else:
        st.write("### 📊 Importation et Synchronisation Cegid (Nouvelle Structure)")
        st.info("Traitement automatique des colonnes 'New Couleur', 'Pointure', 'Dépôt stock' et de la colonne quantité.")

        cegid_files = st.file_uploader("Choisir les fichiers Excel Cegid", type=['xlsx'], accept_multiple_files=True, key="cegid_multi_uploader")

        if cegid_files:
            if st.button("🚀 LANCER LA SYNCHRONISATION DES STOCKS", type="primary"):
                try:
                    conn = pg8000.connect(**DB_CONFIG)
                    cur = conn.cursor()
                    
                    # Ajout de la colonne stock_qty dans le schéma de la table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS cegid_stocks (
                            id SERIAL PRIMARY KEY,
                            product_ref VARCHAR(50) NOT NULL,
                            barcode VARCHAR(50),
                            color VARCHAR(50),
                            size_label VARCHAR(50),
                            depot_stock VARCHAR(100),
                            price NUMERIC(10, 2),
                            stock_qty NUMERIC(10, 2) DEFAULT 0,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    cur.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_cegid_composite 
                        ON cegid_stocks (product_ref, COALESCE(color, ''), COALESCE(size_label, ''), COALESCE(depot_stock, ''));
                    """)
                    conn.commit()

                    all_rows_to_upsert = []
                    for uploaded_file in cegid_files:
                        with st.spinner(f"Lecture de : {uploaded_file.name}..."):
                            df = pd.read_excel(uploaded_file)
                            
                            # Normalisation des noms de colonnes pour éviter les espaces invisibles
                            df.columns = [str(c).strip() for c in df.columns]

                            for index, row in df.iterrows():
                                ref = str(row.get('Code article', '')).strip()
                                if ref == '' or ref == 'nan': continue

                                color = str(row.get('New Couleur', 'N/A')).strip()
                                depot = str(row.get('Dépôt stock', 'Général')).strip()
                                size = str(row.get('Pointure', 'N/A')).strip()
                                price_val = row.get('Prix Détail (TTC)')
                                
                                # --- RECHERCHE DYNAMIQUE DE LA COLONNE STOCK QTY ---
                                # On cherche la valeur numérique juste après le prix si aucun nom de colonne n'est défini
                                try:
                                    price_idx = df.columns.get_loc('Prix Détail (TTC)')
                                    stock_val = row.iloc[price_idx + 1] # Colonne F ou G d'après les images
                                except:
                                    stock_val = 0

                                if color == 'nan' or color == '': color = "N/A"
                                if size == 'nan' or size == '': size = "N/A"
                                if depot == 'nan' or depot == '': depot = "Général"
                                
                                final_price = float(price_val) if pd.notna(price_val) else 0.0
                                final_stock = float(stock_val) if pd.notna(stock_val) else 0.0
                                
                                all_rows_to_upsert.append((ref, None, color, size, depot, final_price, final_stock))

                    if all_rows_to_upsert:
                        with st.spinner(f"Mise à jour de la base de données ({len(all_rows_to_upsert)} lignes)..."):
                            # On met à jour le prix ET la quantité de stock sur conflit composite
                            sql_upsert = """
                                INSERT INTO cegid_stocks (product_ref, barcode, color, size_label, depot_stock, price, stock_qty)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (product_ref, COALESCE(color, ''), COALESCE(size_label, ''), COALESCE(depot_stock, '')) 
                                DO UPDATE SET 
                                    price = EXCLUDED.price,
                                    stock_qty = EXCLUDED.stock_qty,
                                    updated_at = CURRENT_TIMESTAMP;
                            """
                            cur.executemany(sql_upsert, all_rows_to_upsert)
                            conn.commit()
                            st.success(f"✅ Terminé ! {len(all_rows_to_upsert)} lignes synchronisées avec les stocks par dépôt.")
                    
                    conn.close()
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur d'intégration : {e}")