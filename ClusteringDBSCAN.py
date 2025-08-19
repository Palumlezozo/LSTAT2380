from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

def cluster_sunspot_centers_with_dbscan(sunspot_centers, original_image,
                                         eps=80, min_samples=1,
                                         plot_show=False, return_image=False):
    """
    Cluster sunspot centers into distinct groups using DBSCAN and visualize the clusters.

    Args:
    - sunspot_centers: DataFrame containing the x and y coordinates of sunspot centers.
    - original_image: Original solar disk image (grayscale) for visualization.
    - eps: DBSCAN maximum distance between points.
    - min_samples: Minimum number of points to form a cluster.
    - plot_show: Show the annotated image.
    - return_image: If True, return the image with clustered annotations.

    Returns:
    - If return_image=False: DataFrame with cluster labels
    - If return_image=True: Tuple (DataFrame, annotated_image)
    """
    if sunspot_centers.empty:
        print("Aucun centre de tache solaire fourni pour le clustering.")
        if plot_show:
            plt.figure(figsize=(6, 6))
            plt.title("Aucune donnée pour le clustering")
            plt.imshow(original_image, cmap='gray')
            plt.axis('off')
            plt.show()
        empty_df = pd.DataFrame(columns=["x_center", "y_center", "cluster"])
        return (empty_df, original_image) if return_image else empty_df

    # Extraire les coordonnées
    coordinates = sunspot_centers[['x_center', 'y_center']].to_numpy()

    # Appliquer DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)

    # Ajouter les labels au DataFrame
    sunspot_centers['cluster'] = clustering.labels_

    # Préparer l'image annotée
    annotated_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    unique_clusters = sunspot_centers['cluster'].unique()
    colors = np.random.randint(0, 255, size=(len(unique_clusters), 3))

    for _, row in sunspot_centers.iterrows():
        cluster_id = row['cluster']
        x, y = int(row['x_center']), int(row['y_center'])
        color = (255, 255, 255) if cluster_id == -1 else tuple(map(int, colors[cluster_id]))
        cv2.circle(annotated_image, (x, y), 10, color, -1)

    if plot_show:
        plt.figure(figsize=(6, 6))
        plt.title("Clusters des taches solaires")
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.show()

    print(f"Nombre de clusters détectés (hors bruit) : {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
    return (sunspot_centers, annotated_image) if return_image else sunspot_centers
