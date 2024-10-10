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
gdown --fuzzy https://drive.google.com/file/d/1qB48lRCnADnWB24bZMyeASdzFTCEAdQD/view?usp=sharing -O $OUTPUT_DIR/version_drive.txt

if ! cmp -s version.txt version_drive.txt ; then
    gdown --folder https://drive.google.com/drive/folders/1F-84oPjwY6zqReKcOvymW7Q_SfKByVAD?usp=sharing -O $OUTPUT_DIR
    mv $OUTPUT_DIR/version_drive.txt $OUTPUT_DIR/version.txt
fi