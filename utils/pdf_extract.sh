#!/bin/bash

PDF_PATH="$1"
OUTPUT_BASE="${2:-./data/extracted}"  # Default to ./data/extracted if not specified
echo "Processing: $PDF_PATH"
echo "Output to: $OUTPUT_BASE"

# Extract PDF with marker_single
marker_single "$PDF_PATH" --format_lines --redo_inline_math --output_dir ./

# Get the created directory name
PDF_NAME=$(basename "$PDF_PATH" .pdf)

# Find files
EXTRACTED_FILE=$(find "$PDF_NAME" -name "*.md")
JSON_FILE=$(find "$PDF_NAME" -name "*_meta.json")

# Extract title from JSON table_of_contents
TITLE=$(grep -A 5 '"table_of_contents"' "$JSON_FILE" | grep '"title"' | head -1 | sed 's/.*"title": *"//' | sed 's/",$//' | tr '\n' ' ' | sed 's/\\n/ /g')

# Clean title for folder name (remove special chars, limit length)
FOLDER_NAME=$(echo "$TITLE" | sed 's/[^a-zA-Z0-9 ]//g' | sed 's/  */ /g' | cut -c1-50 | sed 's/ *$//')

echo "Title: $TITLE"
echo "Folder: $FOLDER_NAME"

# Create target directory in the output base
mkdir -p "$OUTPUT_BASE/$FOLDER_NAME/images/"

# Move and rename files
mv "$EXTRACTED_FILE" "$OUTPUT_BASE/$FOLDER_NAME/main.md"
mv "$JSON_FILE" "$OUTPUT_BASE/$FOLDER_NAME/main_meta.json"

# Move images
mv "$PDF_NAME"/*page*.* "$OUTPUT_BASE/$FOLDER_NAME/images/" 2>/dev/null

# Clean up
rm -rf "$PDF_NAME"

echo "✓ Done: $OUTPUT_BASE/$FOLDER_NAME/"
echo "✓ Files: main.md, main_meta.json"
echo "✓ Images: images/"
