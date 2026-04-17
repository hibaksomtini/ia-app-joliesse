import os

folder_path = "./catalogue-IA-joliesse"
for filename in os.listdir(folder_path):
    if " " in filename:
        new_name = filename.replace(" ", "-")
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
        print(f"Renommé : {filename} -> {new_name}")