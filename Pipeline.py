import cv2
import pandas as pd
import os
import shutil
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

from DetectCircle import detect_and_normalize_circle
from SunspotsDB import create_sunspot_database
from CalculateCenters import group_sunspots_and_calculate_centers
from PlotCenters import annotate_sunspot_centers
from ClusteringDBSCAN import cluster_sunspot_centers_with_dbscan


PIXEL_THRESHOLD = 3000
recap_data = []
log_erreurs = []
# Interface graphique pour sélectionner plusieurs images
root = tk.Tk()
root.withdraw()

image_paths = filedialog.askopenfilenames(
    title="Sélectionnez les images à analyser",
    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_paths:
    print("Aucun fichier sélectionné.")
    exit()

# Récupérer le dossier parent des images sélectionnées
base_dir = os.path.dirname(image_paths[0])
results_dir = os.path.join(base_dir, "Results")

# Supprimer Results s'il existe, puis le recréer
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)

# Fonction de redimensionnement
def resize_according_to_rules(image, image_name):
    height, width = image.shape[:2]

    if (width, height) == (1110, 1416):
        print(f"{image_name}: déjà au bon format (1110x1416).")
        return image
    elif (width, height) == (1665, 2124) or (0.75 <= width / height <= 0.8):
        print(f"{image_name}: redimensionné en 1110x1416.")
        return cv2.resize(image, (1110, 1416))
    elif width == height:
        print(f"{image_name}: image carrée, redimensionnée à 1200x1200.")
        return cv2.resize(image, (1200, 1200))
    else:
        print(f"{image_name}: format non reconnu — veuillez fournir une image USET ou cropper manuellement.")
        return None

# Boucle sur toutes les images sélectionnées
for path in tqdm(image_paths, desc="Traitement des images") :
    print("\n--- Traitement de :", path, "---")
    image = cv2.imread(path)
    image_name = os.path.basename(path)
    name_no_ext = os.path.splitext(image_name)[0]

    if image is None:
        msg = f"{image_name} — échec du chargement de l image"
        print(msg)
        log_erreurs.append(msg)
        continue

    resized = resize_according_to_rules(image, image_name)
    if resized is None:
        msg = f"{image_name} — format non reconnu ou non supporté"
        print(msg)
        log_erreurs.append(msg)
        continue

    # Créer le sous-dossier Results/<nom_image>
    image_result_dir = os.path.join(results_dir, name_no_ext)
    os.makedirs(image_result_dir, exist_ok=True)

    # Sauvegarder l'image redimensionnée
    resized_path = os.path.join(image_result_dir, "resized.jpg")
    cv2.imwrite(resized_path, resized)

    # Sauvegarder temporairement pour compatibilité avec detect_and_normalize_circle
    temp_path = os.path.join(image_result_dir, "temp_input.jpg")
    cv2.imwrite(temp_path, resized)

    # Pipeline principal
    result = detect_and_normalize_circle(temp_path)
    if result is None:
        msg = f"{image_name} — disque solaire non détecté"
        print(msg)
        log_erreurs.append(msg)
        continue

    cropped_solar_disk, (x_center, y_center), radius = result

    # Étape 1 : base de données des taches
    output_csv_path = os.path.join(image_result_dir, "sunspot_database.csv")
    # Étape 1 : base de données des taches
    result_db = create_sunspot_database(
        cropped_solar_disk,
        (x_center, y_center),
        radius - 12,
        output_csv=output_csv_path,
        plot_show=False,
        return_image=True
    )

    if isinstance(result_db, tuple):
        sunspot_database, visual_db_img = result_db
        if visual_db_img is not None:
            cv2.imwrite(os.path.join(image_result_dir, "sunspots_detected.jpg"), visual_db_img)
        else:
            print(f" Image de base de données non générée pour {image_name}")
    else:
        sunspot_database = result_db
        print(f" Limage annotée de la base de données na pas été retournée pour {image_name}")

    if len(sunspot_database) > PIXEL_THRESHOLD:
        msg = f"{image_name} — disque probablement mal détecté (trop de pixels sombres : {len(sunspot_database)})"
        print("Attention, ", msg)
        log_erreurs.append(msg)
        continue


    # Étape 2 : calcul des centres
    sunspot_centers = group_sunspots_and_calculate_centers(sunspot_database)

    # Étape 3 : annotation des centres
    annotated_img = annotate_sunspot_centers(
        cropped_solar_disk,
        sunspot_centers,
        plot_show=False,
        return_image=True
    )

    if annotated_img is not None:
        cv2.imwrite(os.path.join(image_result_dir, "annotated.jpg"), annotated_img)
    else:
        print(f" L image annotée n a pas été générée pour {image_name}")

    # Étape 4 : clustering DBSCAN
    result_cluster = cluster_sunspot_centers_with_dbscan(
        sunspot_centers,
        cropped_solar_disk,
        plot_show=False,
        return_image=True
    )

    if isinstance(result_cluster, tuple):
        clustered_sunspots, clustered_img = result_cluster
        if clustered_img is not None:
            cv2.imwrite(os.path.join(image_result_dir, "clustered.jpg"), clustered_img)
        else:
            print(f" L image clusterisée n a pas été générée pour {image_name}")
    else:
        clustered_sunspots = result_cluster
        print(f"L image clusterisée n a pas été retournée pour {image_name}")

# --- Récapitulatif de l'image ---
    image_name = os.path.basename(path)

    nb_sunspots = len(sunspot_centers) if sunspot_centers is not None else 0

    if clustered_sunspots is not None and not clustered_sunspots.empty and 'cluster' in clustered_sunspots.columns:
        nb_clusters = len(clustered_sunspots['cluster'].unique())
        if -1 in clustered_sunspots['cluster'].unique():
            nb_clusters -= 1  # retirer le bruit
    else:
        nb_clusters = 0

    recap_data.append({
        "image_traitée": image_name,
        "nb_sunspots": nb_sunspots,
        "nb_clusters": nb_clusters
    })




# Sauvegarde du résumé dans un fichier Excel
recap_df = pd.DataFrame(recap_data)
excel_path = os.path.join(results_dir, "résumé_traitement.xlsx")
recap_df.to_excel(excel_path, index=False)

# Écriture du fichier de log s'il y a eu des erreurs
if log_erreurs:
    log_path = os.path.join(results_dir, "log_erreurs.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Liste des images non traitées :\n\n")
        for ligne in log_erreurs:
            f.write(ligne + "\n")
    print(f"\n Des erreurs ont été rencontrées. Voir le fichier de log : {log_path}")
else:
    print("\n Toutes les images ont été traitées avec succès.")