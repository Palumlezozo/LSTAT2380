PROJECT SUNSPOTS

To execute the code, just execute the pipeline. 
- A Panel to choose the images you want analyze will appear: The results will be stored in "Result" file, in the same location as the images.
- DetectCircle detect and extract the circle from the image
- SunspotsDB creates a database containing every dark pixel ( coordinates and shade value). When executing this, an image appear to make sure the circle is correctly extract and scaned. The radius of the scanned zone is a little smaller than the actual radius so that the algorithm doesn't take its edge for a point to analyse
- CalculateCenters gathers each pixel belonging to the same sunspot, and calculates it's center
- PlotCenters plot on the image to make sure it has the right spots
- ClusteringDBSCAN apply the DBSCAN algorithm from sklearn.cluster to seperate the groups 
