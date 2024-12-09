from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import matplotlib.pyplot as plt

def cluster_sunspot_centers_with_dbscan(sunspot_centers, original_image, eps=40, min_samples=1):
    """
    Cluster sunspot centers into distinct groups using DBSCAN and visualize the clusters.
    
    Args:
    - sunspot_centers: DataFrame containing the x and y coordinates of sunspot centers.
    - original_image: Original solar disk image (grayscale) for visualization.
    - eps: The maximum distance between two samples for them to be considered in the same cluster.
    - min_samples: The minimum number of points required to form a dense region.
    
    Returns:
    - A DataFrame containing the clusters and their respective labels.
    """
    # Extraire les coordonnées des centres des taches
    coordinates = sunspot_centers[['x_center', 'y_center']].to_numpy()

    # Appliquer DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)

    # Ajouter les labels des clusters au DataFrame
    sunspot_centers['cluster'] = clustering.labels_

    # Visualisation des clusters
    annotated_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)  # Convertir l'image en couleur
    unique_clusters = sunspot_centers['cluster'].unique()
    colors = np.random.randint(0, 255, size=(len(unique_clusters), 3))  # Couleurs aléatoires pour les clusters

    for _, row in sunspot_centers.iterrows():
        cluster_id = row['cluster']
        x, y = int(row['x_center']), int(row['y_center'])
        # Assurez-vous que la couleur est convertie en entier
        color = (255, 255, 255) if cluster_id == -1 else tuple(map(int, colors[cluster_id]))
        cv2.circle(annotated_image, (x, y), 10, color, -1)  # Dessiner le point

    # Afficher l'image annotée
    plt.figure(figsize=(6, 6))
    plt.title("Clusters des taches solaires")
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()

    print(f"Nombre de clusters détectés (hors bruit) : {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
    return sunspot_centers

