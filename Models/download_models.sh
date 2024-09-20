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
