#!bin/bash

IMAGE_NAME="Grapes_Extractor_Image"

echo "Monto l'immagine..." 
docker build -t $IMAGE_NAME

echo "Eseguo il container..." 
docker run --gpus all -v /Users/valerio/Desktop/Progetto_Canopies/CanopiesProg./sdg4ad/data/CroppedPatchesDataset:/app/cropped_patches $IMAGE_NAME
