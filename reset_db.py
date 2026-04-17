import pg8000

conn = pg8000.connect(user="postgres", password=r"JoliesseSql**Admin", 
                    host="localhost", database="postgres", port=5433)
cur = conn.cursor()

# Supprime toutes les lignes de la table sans supprimer la table elle-même
cur.execute("TRUNCATE TABLE shoes")
conn.commit()

print("✅ La base de données est maintenant vide.")
cur.close()
conn.close()