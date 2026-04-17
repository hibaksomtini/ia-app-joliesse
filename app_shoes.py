import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageEnhance, ImageTk
import pg8000
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
import threading
import pandas as pd
import requests
from io import BytesIO
from pgvector.pg8000 import register_vector
# --- CONFIGURATION VISUELLE ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ShoeSearchApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Joliesse IA v2 - Permanent Vector Sync")
        self.geometry("950x950")
   
        self.db_config = {
            "user": "postgres.mcmwrchllpqokgcdzmhl", 
            "password": "Joliesse@123456",
            "host": "aws-0-eu-west-1.pooler.supabase.com",
            "database": "postgres",
            "port": 6543 
        }
        self.storage_url = "https://mcmwrchllpqokgcdzmhl.supabase.co/storage/v1/object/public/catalogue/"
        
        print("Initialisation de l'IA (CLIP ViT-L-14)...")
        self.model = SentenceTransformer('clip-ViT-L-14')
        self.setup_ui()

    def setup_ui(self):
        self.header = ctk.CTkFrame(self, height=80, fg_color="#1f538d", corner_radius=0)
        self.header.pack(fill="x", side="top")
        self.title_label = ctk.CTkLabel(self.header, text="JOLIESSE IA PRO", font=("Roboto", 28, "bold"), text_color="white")
        self.title_label.place(relx=0.5, rely=0.5, anchor="center")

        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.pack(pady=30, padx=30, fill="x")

        self.btn_select = ctk.CTkButton(self.control_frame, text="✂️ RECADRER & CHERCHER", 
                                        command=self.select_image, height=70, 
                                        font=("Roboto", 18, "bold"), corner_radius=15)
        self.btn_select.pack(side="left", expand=True, padx=10)

        self.btn_full_search = ctk.CTkButton(self.control_frame, text="🔍 IMAGE ENTIÈRE", 
                                             command=self.search_full_image, height=70,
                                             font=("Roboto", 18, "bold"), corner_radius=15)
        self.btn_full_search.pack(side="left", expand=True, padx=10)

        self.btn_view = ctk.CTkButton(self.control_frame, text="📦 CATALOGUE", 
                                     command=self.open_catalog_viewer, height=70,
                                     fg_color="#3d3d3d", font=("Roboto", 14), corner_radius=15)
        self.btn_view.pack(side="left", expand=True, padx=10)

        self.preview_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.preview_frame.pack(pady=10)

        self.img_preview = ctk.CTkLabel(self.preview_frame, text="Prêt pour la recherche", 
                                        width=300, height=300, fg_color="#2b2b2b", 
                                        corner_radius=15)
        self.img_preview.pack()

        self.results_frame = ctk.CTkScrollableFrame(self, width=800, height=500, fg_color="#1a1a1a")
        self.results_frame.pack(pady=20, padx=20, fill="both", expand=True)

    def process_image_pro(self, pil_img):
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGBA")
            background = Image.new("RGBA", pil_img.size, (255, 255, 255))
            pil_img = Image.alpha_composite(background, pil_img).convert("RGB")
        else:
            pil_img = pil_img.convert("RGB")
        
        desired_size = 224
        pil_img.thumbnail((desired_size, desired_size), Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (desired_size, desired_size), (255, 255, 255))
        new_img.paste(pil_img, ((desired_size - pil_img.size[0]) // 2, (desired_size - pil_img.size[1]) // 2))
        return new_img

    def get_embedding(self, img_path):
        img = Image.open(img_path)
        return self.get_embedding_from_pil(img)

    def get_embedding_from_pil(self, pil_img):
        processed = self.process_image_pro(pil_img)
        embedding = self.model.encode(processed, convert_to_numpy=True)
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist()

    def run_search(self, path):
        self.after(0, self.clear_results)
        try:
            conn = pg8000.connect(**self.db_config)
            register_vector(conn)
            cur = conn.cursor()

            # Vecteur de la recherche
            query_vec = np.array(self.get_embedding(path)).flatten()

            cur.execute("SELECT product_ref, price, image_paths, embedding, colors FROM products")
            rows = cur.fetchall()

            results = []
            for ref, price, img_data, db_emb, colors in rows:
                if db_emb is None or len(db_emb) != 768:
                    try:
                        image_name = str(img_data).split('|')[0].strip()
                        url = self.storage_url + image_name
                        resp = requests.get(url, timeout=5)
            
                        if resp.status_code == 200:
                            img = Image.open(BytesIO(resp.content))
                            db_emb_list = self.get_embedding_from_pil(img)
                
                            # 1. On formate bien avec les crochets []
                            formatted_vector = "[" + ",".join(map(str, db_emb_list)) + "]"
                
                            # 2. On tente l'update
                            cur.execute("UPDATE products SET embedding = %s WHERE product_ref = %s", (formatted_vector, ref))
                            conn.commit() # Si ça marche, on valide
                
                            db_emb = db_emb_list
                            print(f"✅ Vecteur réparé pour : {ref}")
                        else:
                            continue

                    except Exception as e:
                        # --- LE POINT CRUCIAL ICI ---
                        conn.rollback() # <--- C'est cette ligne qui débloque la transaction !
                        print(f"⚠️ Échec réparation {ref}: {e}")
                        continue

                # Calcul de similarité
                db_vec = np.array(db_emb).flatten()
                score = np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec))
                
                img_list = str(img_data).split('|') if img_data else []
                results.append({"ref": ref, "score": float(score), "price": price, "images": img_list, "colors": colors})

            results.sort(key=lambda x: x["score"], reverse=True)
            
            for res in results[:8]:
                # Ligne 152 (dans run_search)
                self.after(0, lambda r=res: self.create_result_card(
                    r.get("ref", "Inconnu"), 
                    r.get("score", 0), 
                    r.get("price", 0), 
                    r.get("colors", ""), 
                    r.get("images", [])
                ))
            
            self.after(0, lambda: self.btn_full_search.configure(text="🔍 IMAGE ENTIÈRE", state="normal"))
            conn.close()
        except Exception as e: 
            print(f"❌ Erreur recherche : {e}")

    def create_result_card(self, ref, score, price=0.0, colors=None, image_list=None):
        card = ctk.CTkFrame(self.results_frame, fg_color="#252525", border_width=1, border_color="#3d3d3d")
        card.pack(fill="x", pady=8, padx=10)
        
        img_label = ctk.CTkLabel(card, text="⌛", width=130, height=130)
        img_label.pack(side="left", padx=10, pady=10)

        def load_image():
            if image_list:
                url = self.storage_url + image_list[0].strip()
                try:
                    res = requests.get(url, timeout=5)
                    if res.status_code == 200:
                        img_data = Image.open(BytesIO(res.content))
                        ctk_img = ctk.CTkImage(light_image=img_data, size=(130, 130))
                        img_label.configure(image=ctk_img, text="")
                except: img_label.configure(text="Erreur")

        threading.Thread(target=load_image, daemon=True).start()

        txt_frame = ctk.CTkFrame(card, fg_color="transparent")
        txt_frame.pack(side="left", padx=20, fill="both", expand=True)
        ctk.CTkLabel(txt_frame, text=ref, font=("Roboto", 18, "bold")).pack(anchor="w")
        # Remplacez l'ancienne ligne du prix par celle-ci :
        display_price = f"{price:.2f}" if price is not None else "0.00"
        ctk.CTkLabel(txt_frame, text=f"Prix: {display_price} DT", text_color="#27ae60", font=("Roboto", 16)).pack(anchor="w")
        
        confidence = score * 100
        ctk.CTkLabel(txt_frame, text=f"Match: {confidence:.2f}%", text_color="#f1c40f").pack(anchor="w")

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if file_path:
            self.open_crop_tool(file_path)

    def search_full_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if file_path:
            img = Image.open(file_path)
            self.img_preview.configure(image=ctk.CTkImage(light_image=img, size=(280, 280)), text="")
            self.btn_full_search.configure(text="⌛ Analyse...", state="disabled")
            threading.Thread(target=self.run_search, args=(file_path,), daemon=True).start()

    def clear_results(self):
        for widget in self.results_frame.winfo_children(): widget.destroy()

    def open_crop_tool(self, image_path):
        crop_window = ctk.CTkToplevel(self)
        crop_window.title("Recadrer l'article")
        crop_window.geometry("800x950")
        crop_window.attributes("-topmost", True)  # Garde la fenêtre au premier plan
        crop_window.grab_set()

        # Cadre pour le bouton (fixe en haut pour ne pas le perdre)
        top_bar = ctk.CTkFrame(crop_window, height=60)
        top_bar.pack(fill="x", side="top", padx=10, pady=10)

        original_pil = Image.open(image_path).convert("RGB")
        
        # Redimensionnement intelligent pour l'affichage
        display_width = 700
        ratio = original_pil.width / display_width
        display_h = int(original_pil.height / ratio)
        
        img_display = original_pil.resize((display_width, display_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_display)

        self.crop_data = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "active": False}
        
        canvas = ctk.CTkCanvas(crop_window, width=display_width, height=display_h, highlightthickness=0)
        canvas.pack(pady=10)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas.image = tk_img 

        rect_id = [None]

        def on_press(e):
            self.crop_data["x1"], self.crop_data["y1"] = e.x, e.y
            self.crop_data["active"] = True
            if rect_id[0]: canvas.delete(rect_id[0])
            rect_id[0] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline='red', width=3)

        def on_drag(e):
            if self.crop_data["active"]:
                self.crop_data["x2"], self.crop_data["y2"] = e.x, e.y
                canvas.coords(rect_id[0], self.crop_data["x1"], self.crop_data["y1"], e.x, e.y)

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)

        def validate():
            if self.crop_data["x2"] == 0: # Si l'utilisateur n'a pas glissé
                messagebox.showwarning("Attention", "Veuillez dessiner un rectangle sur la chaussure")
                return
                
            left = min(self.crop_data["x1"], self.crop_data["x2"]) * ratio
            top = min(self.crop_data["y1"], self.crop_data["y2"]) * ratio
            right = max(self.crop_data["x1"], self.crop_data["x2"]) * ratio
            bottom = max(self.crop_data["y1"], self.crop_data["y2"]) * ratio
            
            cropped = original_pil.crop((left, top, right, bottom))
            cropped.save("temp_crop.jpg")
            
            # Mise à jour de l'aperçu sur la fenêtre principale
            self.img_preview.configure(image=ctk.CTkImage(light_image=cropped, size=(280, 280)), text="")
            
            crop_window.destroy() # Ferme la fenêtre de crop
            
            # Lance la recherche
            self.btn_full_search.configure(text="⌛ Analyse du crop...", state="disabled")
            threading.Thread(target=self.run_search, args=("temp_crop.jpg",), daemon=True).start()

        # Bouton placé dans la barre du haut pour être toujours visible
        ctk.CTkButton(top_bar, text="✅ LANCER LA RECHERCHE", 
                      fg_color="#27ae60", hover_color="#219150",
                      command=validate, font=("Roboto", 14, "bold")).pack(expand=True, pady=10)

    def open_catalog_viewer(self):
        viewer = ctk.CTkToplevel(self)
        viewer.title("Gestion Catalogue Joliesse IA")
        viewer.geometry("700x900")
        viewer.attributes("-topmost", True)

        # --- BARRE DE RECHERCHE ---
        search_frame = ctk.CTkFrame(viewer, fg_color="transparent")
        search_frame.pack(fill="x", padx=20, pady=15)

        search_entry = ctk.CTkEntry(search_frame, placeholder_text="🔍 Rechercher une référence (ex: 4414...)", height=40)
        search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # --- ZONE DE LISTE ---
        scroll = ctk.CTkScrollableFrame(viewer, fg_color="#1a1a1a")
        scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        def refresh_list(filter_text=""):
            # Nettoyer la liste actuelle
            for widget in scroll.winfo_children():
                widget.destroy()

            try:
                conn = pg8000.connect(**self.db_config)
                cur = conn.cursor()
                
                # SQL : Filtrer si du texte est saisi
                if filter_text:
                    query = "SELECT product_ref, price, image_paths FROM products WHERE product_ref ILIKE %s ORDER BY product_ref ASC LIMIT 50"
                    cur.execute(query, (f"%{filter_text}%",))
                else:
                    cur.execute("SELECT product_ref, price, image_paths FROM products ORDER BY product_ref ASC LIMIT 50")
                
                rows = cur.fetchall()
                
                if not rows:
                    ctk.CTkLabel(scroll, text="Aucun résultat trouvé").pack(pady=20)
                
                for ref, price, img_data in rows:
                    row_frame = ctk.CTkFrame(scroll, fg_color="#2b2b2b", height=100)
                    row_frame.pack(fill="x", pady=5, padx=5)

                    # 1. Miniature
                    img_label = ctk.CTkLabel(row_frame, text="⌛", width=80, height=80, corner_radius=8, fg_color="#222")
                    img_label.pack(side="left", padx=15, pady=10)

                    # 2. Infos
                    info_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
                    info_frame.pack(side="left", fill="both", expand=True)
                    
                    ctk.CTkLabel(info_frame, text=f"REF: {ref}", font=("Roboto", 16, "bold")).pack(anchor="w", pady=(15, 0))
                    
                    p_val = f"{price:.2f} DT" if price is not None else "--- DT"
                    ctk.CTkLabel(info_frame, text=p_val, text_color="#27ae60", font=("Roboto", 14)).pack(anchor="w")

                    # Thread de chargement d'image
                    def load_img(label=img_label, data=img_data):
                        if data:
                            try:
                                img_name = str(data).split('|')[0].strip()
                                url = self.storage_url + img_name
                                res = requests.get(url, timeout=3)
                                if res.status_code == 200:
                                    img_pil = Image.open(BytesIO(res.content))
                                    img_pil.thumbnail((80, 80))
                                    ctk_img = ctk.CTkImage(light_image=img_pil, size=(80, 80))
                                    label.after(0, lambda: label.configure(image=ctk_img, text=""))
                            except:
                                label.after(0, lambda: label.configure(text="❌"))

                    threading.Thread(target=load_img, daemon=True).start()

                conn.close()
            except Exception as e:
                print(f"Erreur catalogue: {e}")

        # Événement de recherche (touche Entrée ou frappe)
        search_entry.bind("<Return>", lambda e: refresh_list(search_entry.get()))
        
        # Bouton pour lancer la recherche manuellement
        btn_search = ctk.CTkButton(search_frame, text="Filtrer", width=80, command=lambda: refresh_list(search_entry.get()))
        btn_search.pack(side="right")

        # Chargement initial
        refresh_list()

if __name__ == "__main__":
    app = ShoeSearchApp()
    app.mainloop()