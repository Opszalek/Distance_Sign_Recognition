#!/bin/bash

# Install gdown if not already installed
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found, installing..."
    pip install gdown
fi

# Output directory
OUTPUT_DIR="."

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Download the folder
gdown --folder https://drive.google.com/drive/folders/1F-84oPjwY6zqReKcOvymW7Q_SfKByVAD -O $OUTPUT_DIR

for dir in "$OUTPUT_DIR"/*/; do
    if [ -d "$dir" ]; then
        cd $dir

        for file in *.zip; do
            echo "Processing file: $file"
            if [ -f "$file" ]; then
                dirname=$(basename "$file" .zip)
                mkdir -p "$dirname"
                unzip -o "$file" -d "$dirname"
                rm "$file"
            fi
        done
        cd ../..
    fi
done