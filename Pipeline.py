from DetectCircle import detect_and_normalize_circle
from DetectSunspots import detect_sunspots
from SunspotsDB import create_sunspot_database
from CalculateCenters import group_sunspots_and_calculate_centers
from PlotCenters import annotate_sunspot_centers

# Pipeline

image_path = r"C:\Users\quent\Desktop\UCL\LSTAT2380\USETSample24Unannotated\Picture2.jpg"


# Appeler detect_and_normalize_circle
cropped_solar_disk, (x_center, y_center), radius = detect_and_normalize_circle(image_path)

if cropped_solar_disk is not None:
    # Étape 2 : Analyser les pixels sombres et créer une base de données
    output_csv_path = "sunspot_database.csv"  # Chemin de sauvegarde du fichier CSV
    sunspot_database = create_sunspot_database(cropped_solar_disk, (x_center, y_center), radius-5, output_csv=output_csv_path)

    # Vérifier les premières lignes
    print(sunspot_database.head())

    sunspot_centers = group_sunspots_and_calculate_centers(sunspot_database)
    print(sunspot_centers.head())

    annotated_image = annotate_sunspot_centers(cropped_solar_disk, sunspot_centers)
else:
    print("Le disque solaire n'a pas été détecté.")




