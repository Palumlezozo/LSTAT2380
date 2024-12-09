PROJECT SUNSPOTS

To execute the code, just execute the pipeline. 
- DetectCircle detect and extract the circle from the image
- SunspotsDB creates a database containing every dark pixel ( coordinates and shade value)
- CalculateCenters gathers each pixel belonging to the same sunspot, and calculates it's center
- PlotCenters plot on the image to make sure it has the right spots 
