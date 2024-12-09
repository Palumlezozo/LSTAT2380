import cv2
import numpy as np
import matplotlib.pyplot as plt

def annotate_sunspot_centers(image, sunspot_centers):
    """
    Annotate the centers of sunspots on the image.
    
    Args:
    - image: Original solar disk image (grayscale).
    - sunspot_centers: DataFrame containing the x and y coordinates of the sunspot centers.
    
    Returns:
    - Annotated image with sunspot centers marked in green.
    """
    # Convertir l'image en couleur pour permettre l'annotation
    annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Ajouter des cercles verts aux positions des centres des taches solaires
    for _, row in sunspot_centers.iterrows():
        x, y = int(row['x_center']), int(row['y_center'])
        cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)  # Cercle vert de rayon 5

    # Afficher l'image annotée
    plt.figure(figsize=(6, 6))
    plt.title("Centres des taches solaires")
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()

    return annotated_image
