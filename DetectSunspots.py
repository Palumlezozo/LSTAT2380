from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_sunspots(solar_disk, disk_center, disk_radius):
    """
    Detect sunspots within the solar disk (inside the circular region).
    Returns the centers of sunspots.
    """
    # Étape 1 : Créer un masque circulaire avec un rayon ajusté
    adjusted_radius = disk_radius - 5  # Réduire légèrement le rayon pour éviter les bords
    mask = np.zeros_like(solar_disk, dtype=np.uint8)
    cv2.circle(mask, disk_center, adjusted_radius, 255, -1)  # Masque circulaire rempli

    # Appliquer le masque pour ne conserver que l'intérieur du cercle
    masked_disk = cv2.bitwise_and(solar_disk, solar_disk, mask=mask)

    # Étape 2 : Trouver les pixels noirs (intensité faible)
    threshold = 100  # Définir un seuil pour les pixels sombres
    binary = (masked_disk < threshold).astype(np.uint8) * 255  # Crée une image binaire

    # Étape 3 : Trouver les contours des clusters de pixels noirs
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Étape 4 : Calculer les centres de chaque cluster
    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 5:  # Exclure les très petits clusters
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Vérifier si le centre est bien dans le disque solaire
                distance_to_center = ((cX - disk_center[0])**2 + (cY - disk_center[1])**2)**0.5
                if distance_to_center < adjusted_radius:
                    centers.append((cX, cY))

    # Étape 5 : Visualisation
    output = cv2.cvtColor(solar_disk, cv2.COLOR_GRAY2BGR)
    for center in centers:
        cv2.circle(output, center, 10, (0, 255, 0), -1)  # Dessiner un point pour chaque centre

    print(f"Nombre de taches solaires détectées : {len(centers)}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Disque solaire masqué")
    plt.imshow(masked_disk, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Taches solaires détectées")
    plt.imshow(output)
    plt.axis('off')
    plt.show()

    return centers


"""def detect_sunspots(solar_disk):

    # Step 1: Apply addaptive threshold to detect dark pixels
    _, binary = cv2.threshold(solar_disk, 10, 255, cv2.THRESH_BINARY_INV)

    # Step 2 : Find contours oh dark zones
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 3 : Filter contours by size to avoid noize
    min_area = 20  # Minimal air to be considered as a sunspot
    sunspots = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Step 4 : Draw a copy of the disk with the contours
    output = cv2.cvtColor(solar_disk, cv2.COLOR_GRAY2BGR)
    for cnt in sunspots:
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)  

    # Step 5 : Print results
    print(f"Number of sunspots counted : {len(sunspots)}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original solar disk")
    plt.imshow(solar_disk, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Sunspots detected")
    plt.imshow(output)
    plt.axis('off')

    plt.show()

    return sunspots
"""