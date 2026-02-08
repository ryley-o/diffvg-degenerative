#!/bin/bash

# Batch script to run painterly_rendering_custom_strokes_gpu.py on multiple images
# GPU REQUIRED - This script will fail if CUDA GPU is not available
# Usage: ./run_painterly_strokes_batch_gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA GPU may not be available."
    echo "This script REQUIRES a CUDA-capable GPU."
    exit 1
fi

# Check if GPU is available via Python
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "ERROR: PyTorch CUDA is not available!"
    echo "This script REQUIRES a CUDA-capable GPU with PyTorch CUDA support."
    exit 1
fi

# Display GPU information
echo "=========================================="
echo "GPU Batch Processing Script"
echo "=========================================="
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

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
    
    # Run GPU version
    python3 painterly_rendering_custom_strokes_gpu.py --input_filename "$IMAGE_FILE" \
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
echo "GPU Batch processing complete!"
echo "=========================================="
