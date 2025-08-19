import cv2
import numpy as np
import matplotlib.pyplot as plt

def annotate_sunspot_centers(image, sunspot_centers, plot_show=False, return_image=False):
    """
    Annotate the centers of sunspots on the image.

    Args:
    - image: Original solar disk image (grayscale).
    - sunspot_centers: DataFrame containing the x and y coordinates of the sunspot centers.
    - plot_show: Display the image with annotations (optional).
    - return_image: If True, return the annotated image explicitly.

    Returns:
    - If return_image is True: the annotated image (BGR)
    - If False: nothing (the display is handled by plot_show)
    """
    if sunspot_centers.empty:
        print("Aucun centre de tache solaire détecté pour l'annotation.")
        if plot_show:
            plt.figure(figsize=(6, 6))
            plt.title("Aucun centre détecté")
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.show()
        return image if return_image else None

    # Convertir en couleur pour annotation
    annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for _, row in sunspot_centers.iterrows():
        x, y = int(row['x_center']), int(row['y_center'])
        cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)  # Cercle vert

    if plot_show:
        plt.figure(figsize=(6, 6))
        plt.title("Centres des taches solaires")
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.show()

    return annotated_image if return_image else None

