#!/bin/bash

# Batch script to run painterly_rendering_custom_strokes.py on multiple images
# Usage: ./run_painterly_strokes_batch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parameters
NUM_PATHS=2048
NUM_ITER=650
MAX_WIDTH=4.0

# Process images - adjust the range as needed
# Example: processing _08.JPG, _09.JPG, etc.
for i in {0..41}; do
    # Format with leading zero for single digits
    if [ $i -lt 10 ]; then
        IMAGE_FILE="_0${i}.JPG"
    else
        IMAGE_FILE="_${i}.JPG"
    fi
    
    # Check if file exists before processing
    if [ ! -f "source_img/$IMAGE_FILE" ]; then
        echo "Skipping $IMAGE_FILE (file not found)"
        continue
    fi
    
    echo "=========================================="
    echo "Processing: $IMAGE_FILE"
    echo "=========================================="
    
    python painterly_rendering_custom_strokes.py --input_filename "$IMAGE_FILE" \
        --num_paths "$NUM_PATHS" \
        --num_iter "$NUM_ITER" \
        --max_width "$MAX_WIDTH"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $IMAGE_FILE"
    else
        echo "✗ Error processing: $IMAGE_FILE"
        echo "Continuing with next image..."
    fi
    
    echo ""
done

echo "=========================================="
echo "Batch processing complete!"
echo "=========================================="

