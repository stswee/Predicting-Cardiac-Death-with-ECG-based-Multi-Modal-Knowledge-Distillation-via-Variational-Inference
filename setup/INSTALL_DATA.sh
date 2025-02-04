#!/bin/bash

#!/bin/bash

# Define the destination folder
DEST_FOLDER="../Holter_ECG_Dataset"

# Create the folder if it doesn't exist
mkdir -p "$DEST_FOLDER"

# Base URL of the dataset
BASE_URL="https://physionet.org/static/published-projects/music-sudden-cardiac-death/1.0.1/Holter_ECG/"

# Get a list of files in the Holter_ECG directory
FILE_LIST=$(curl -s "https://physionet.org/content/music-sudden-cardiac-death/1.0.1/Holter_ECG/" | grep -oP '(?<=href=")[^"]*' | grep -E '^[^/]+$')

# Download each file
for FILE in $FILE_LIST; do
    echo "Downloading $FILE..."
    wget -q --show-progress -P "$DEST_FOLDER" "${BASE_URL}${FILE}"
done
