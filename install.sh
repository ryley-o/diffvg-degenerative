#!/bin/bash
# Installation script for diffvg on a remote server with CUDA GPU
# Usage: ./install.sh

set -e  # Exit on error

echo "=========================================="
echo "diffvg Installation Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    print_error "setup.py not found. Please run this script from the diffvg root directory."
    exit 1
fi

# Step 1: Check prerequisites
print_info "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "python3 not found. Please install Python 3.7+ first."
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_info "Found Python: $PYTHON_VERSION"

# Check pip
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    print_error "pip not found. Please install pip first."
    exit 1
fi
PIP_CMD=$(command -v pip3 || command -v pip)
print_info "Found pip: $PIP_CMD"

# Check git
if ! command -v git &> /dev/null; then
    print_error "git not found. Please install git first."
    exit 1
fi
print_info "Found git: $(git --version)"

# Check cmake
if ! command -v cmake &> /dev/null; then
    print_warn "cmake not found. Will try to install via pip/conda."
else
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    print_info "Found cmake: $CMAKE_VERSION"
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_info "Found CUDA: $CUDA_VERSION"
else
    print_warn "nvcc not found in PATH. CUDA support may not be available."
fi

# Check PyTorch
print_info "Checking for PyTorch..."
if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    print_info "Found PyTorch: $TORCH_VERSION"
    
    # Check if PyTorch has CUDA support
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        CUDA_AVAILABLE=$(python3 -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')")
        print_info "PyTorch CUDA available: $CUDA_AVAILABLE"
        if [ "$CUDA_AVAILABLE" = "Yes" ]; then
            GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
            print_info "Number of GPUs detected: $GPU_COUNT"
        fi
    else
        print_warn "PyTorch CUDA not available. Building without CUDA support."
    fi
else
    print_warn "PyTorch not found. Will attempt to install."
    print_info "Installing PyTorch with CUDA support..."
    $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cu118 || \
    $PIP_CMD install torch torchvision
fi

echo ""

# Step 2: Initialize git submodules
print_info "Initializing git submodules..."
if git submodule update --init --recursive; then
    print_info "Git submodules initialized successfully."
else
    print_error "Failed to initialize git submodules."
    exit 1
fi
echo ""

# Step 3: Install Python dependencies
print_info "Installing Python dependencies..."

# Check if conda is available
if command -v conda &> /dev/null; then
    print_info "Using conda for some dependencies..."
    conda install -y numpy scikit-image cmake ffmpeg -c conda-forge || \
    print_warn "Some conda packages failed, continuing with pip..."
fi

# Install via pip
print_info "Installing packages via pip..."
$PIP_CMD install --upgrade pip setuptools wheel
$PIP_CMD install svgwrite svgpathtools cssutils numba torch-tools visdom scikit-image

# Install cmake if not available
if ! command -v cmake &> /dev/null; then
    print_info "Installing cmake via pip..."
    $PIP_CMD install cmake || print_warn "Failed to install cmake via pip. Please install cmake manually."
fi

echo ""

# Step 4: Verify dependencies
print_info "Verifying dependencies..."
MISSING_DEPS=()
for dep in numpy skimage svgwrite svgpathtools cssutils numba ttools visdom; do
    if ! python3 -c "import $dep" 2>/dev/null; then
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    print_warn "Missing dependencies: ${MISSING_DEPS[*]}"
    print_info "Attempting to install missing dependencies..."
    for dep in "${MISSING_DEPS[@]}"; do
        case $dep in
            skimage)
                $PIP_CMD install scikit-image
                ;;
            ttools)
                $PIP_CMD install torch-tools
                ;;
            *)
                $PIP_CMD install "$dep"
                ;;
        esac
    done
fi

echo ""

# Step 5: Build and install diffvg
print_info "Building and installing diffvg (this may take several minutes)..."

# Force CUDA build if CUDA is available
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    print_info "Building with CUDA support..."
    export DIFFVG_CUDA=1
else
    print_info "Building without CUDA support..."
    export DIFFVG_CUDA=0
fi

# Build and install
if python3 setup.py install; then
    print_info "diffvg installed successfully!"
else
    print_error "Failed to build and install diffvg."
    exit 1
fi

echo ""

# Step 6: Verify installation
print_info "Verifying installation..."
if python3 -c "import pydiffvg; print('pydiffvg version:', pydiffvg.__version__ if hasattr(pydiffvg, '__version__') else 'installed')" 2>/dev/null; then
    print_info "✓ pydiffvg imported successfully"
    
    # Check CUDA availability in pydiffvg
    if python3 -c "import pydiffvg, torch; device = pydiffvg.get_device(); print('Device:', device)" 2>/dev/null; then
        DEVICE=$(python3 -c "import pydiffvg, torch; device = pydiffvg.get_device(); print(device)")
        print_info "✓ pydiffvg device: $DEVICE"
    fi
else
    print_error "Failed to import pydiffvg. Installation may have failed."
    exit 1
fi

echo ""
echo "=========================================="
print_info "Installation completed successfully!"
echo "=========================================="
echo ""
print_info "You can now run scripts like:"
echo "  cd apps"
echo "  python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0"
echo ""
