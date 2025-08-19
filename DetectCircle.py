import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_normalize_circle(image_path, plot_show = False):
    """
    Detect the solar disk in the image and normalize it.
    Returns the cropped disk, its center, and its radius.
    """
    # Loading image in gray levels
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None

    # Reduce noise
    blurred = cv2.GaussianBlur(image, (11, 11), 0)

    # Circle Detection
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                               param1=50, param2=30, minRadius=200, maxRadius=800)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]  # Take the first circle detected
        print(f"Cercle detected : Centre=({x}, {y}), Radius={r}")

        # Extract solar disk
        mask = np.ones_like(image, dtype=np.uint8) * 255  # Masque initialisé à blanc
        cv2.circle(mask, (x, y), r, 0, -1)  # Dessiner le cercle en noir (intérieur du disque)

        # Remplacer les pixels en dehors du cercle par du blanc
        solar_disk = cv2.bitwise_or(image, mask)

        cropped = solar_disk[y-r:y+r, x-r:x+r]

        # Print results
        if plot_show:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original image")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Solar Disk")
            plt.imshow(cropped, cmap='gray')
            plt.axis('off')
            plt.show()

        return cropped,(r, r), r  # Retourne l'image, le centre, et le rayon

    else:
        print("No circle detected")
        return None, None, None
