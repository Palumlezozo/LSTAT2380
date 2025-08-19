import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_sunspot_database(solar_disk, center, radius, intensity_threshold=170, output_csv="sunspot_database.csv", plot_show = False, return_image=False):
    """
    Analyze pixels within a circle and create a database of pixels with intensity below a threshold.
    Also, visualize the scanned circle and save the database to a CSV file.
    
    Args:
    - solar_disk: Grayscale image of the solar disk.
    - center: Tuple (x, y) representing the center of the circle.
    - radius: Radius of the circle.
    - intensity_threshold: Intensity threshold for detecting dark pixels (default = 160).
    - output_csv: Path to save the output CSV file (default = "sunspot_database.csv").
    
    Returns:
    - DataFrame containing positions and intensities of dark pixels.
    """
    # Dimensions de l'image
    height, width = solar_disk.shape
    x_center, y_center = center

    # Étape 1 : Visualisation du cercle scanné
    annotated_image = cv2.cvtColor(solar_disk, cv2.COLOR_GRAY2BGR)  # Convertir en RGB pour dessiner en couleur
    cv2.circle(annotated_image, (int(x_center), int(y_center)), int(radius), (0, 255, 0), 2)  # Cercle vert


    if plot_show: 
        plt.figure(figsize=(6, 6))
        plt.title("Zone scannée")
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.show()

    # Liste pour stocker les données des pixels
    pixel_data = []

    # Étape 2 : Parcourir chaque pixel de l'image
    for y in range(height):
        for x in range(width):
            # Calculer la distance au centre du cercle
            distance_to_center = ((x - x_center)**2 + (y - y_center)**2)**0.5
            
            # Vérifier si le pixel est à l'intérieur du cercle
            if distance_to_center <= radius:
                # Vérifier si l'intensité du pixel est inférieure au seuil
                intensity = solar_disk[y, x]
                if intensity < intensity_threshold:
                    pixel_data.append({"x": x, "y": y, "intensity": intensity})

    # Étape 3 : Créer un DataFrame à partir des données collectées
    if pixel_data:
        df = pd.DataFrame(pixel_data)
        print(f"Nombre total de pixels détectés avec une intensité < {intensity_threshold}: {len(df)}")
    else:
        df = pd.DataFrame(columns=["x", "y", "intensity"])
        print("Aucune tache solaire détectée.")

    # Étape 4 : Exporter le DataFrame en fichier CSV
    df.to_csv(output_csv, index=False)
    print(f"Base de données sauvegardée dans le fichier : {output_csv}")

    print(f"Nombre total de pixels détectés avec une intensité < {intensity_threshold}: {len(df)}")
    if return_image:
        return df, annotated_image
    else:
        return df
