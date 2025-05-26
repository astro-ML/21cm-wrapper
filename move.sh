#!/bin/bash

# Define source directories
DIR0="data_sbi_2_ext"
DIR1="data_sbi_1_ext"
DIR2="data_sbi_3_ext"
TARGET_DIR="data_sbi_noTS"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Move and rename files from data_sbi_0
if [ -d "$DIR0" ]; then
  for file in "$DIR0"/*; do
    if [ -f "$file" ]; then
      filename=$(basename "$file")
      mv "$file" "$TARGET_DIR/71_$filename"
    fi
  done
else
  echo "Directory $DIR0 does not exist."
fi

# Move and rename files from data_sbi_1
if [ -d "$DIR1" ]; then
  for file in "$DIR1"/*; do
    if [ -f "$file" ]; then
      filename=$(basename "$file")
      mv "$file" "$TARGET_DIR/72_$filename"
    fi
  done
else
  echo "Directory $DIR1 does not exist."
fi

# Move and rename files from data_sbi_1
if [ -d "$DIR2" ]; then
  for file in "$DIR2"/*; do
    if [ -f "$file" ]; then
      filename=$(basename "$file")
      mv "$file" "$TARGET_DIR/73_$filename"
    fi
  done
else
  echo "Directory $DIR2 does not exist."
fi

echo "Files have been moved and renamed into $TARGET_DIR."
