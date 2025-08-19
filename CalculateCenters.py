import pandas as pd
import numpy as np


def group_sunspots_and_calculate_centers(sunspot_database):
    """
    Group sunspot pixels into distinct sunspots based on adjacency and calculate the centers of each sunspot.
    
    Args:
    - sunspot_database: DataFrame containing positions and intensities of dark pixels.
    
    Returns:
    - A DataFrame containing the sunspots and their calculated centers.
    """
    # Convertir la base de données en un numpy array pour traitement rapide
    pixels = sunspot_database[['x', 'y']].to_numpy()

    # Structure pour stocker les groupes de pixels
    groups = []
    visited = set()

    # Fonction pour regrouper les pixels adjacents
    def dfs(pixel, group):
        stack = [pixel]
        while stack:
            current = stack.pop()
            if tuple(current) in visited:
                continue
            visited.add(tuple(current))
            group.append(current)
            # Rechercher les pixels adjacents
            for neighbor in pixels:
                if tuple(neighbor) not in visited:
                    if abs(current[0] - neighbor[0]) <= 1 and abs(current[1] - neighbor[1]) <= 1:
                        stack.append(neighbor)

    # Regrouper les pixels en taches
    for pixel in pixels:
        if tuple(pixel) not in visited:
            group = []
            dfs(pixel, group)
            groups.append(group)

    # Calculer le centre de chaque tache
    sunspot_centers = []
    for group in groups:
        group = np.array(group)
        x_center = (group[:, 0].min() + group[:, 0].max()) // 2
        y_center = (group[:, 1].min() + group[:, 1].max()) // 2
        sunspot_centers.append({"x_center": x_center, "y_center": y_center, "size": len(group)})

    # Créer un DataFrame pour les centres des taches
    centers_df = pd.DataFrame(sunspot_centers)
    print(f"Nombre total de taches solaires détectées : {len(centers_df)}")
    return centers_df
