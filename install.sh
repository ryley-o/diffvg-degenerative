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

# Update pybind11 to a version compatible with Python 3.12
# Python 3.12 requires pybind11 >= 2.11.1 (current submodule is 2.6.0)
print_info "Checking Python version for pybind11 compatibility..."
PYTHON_VERSION_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "3")
PYTHON_VERSION_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")

if [ "$PYTHON_VERSION_MAJOR" -eq 3 ] && [ "$PYTHON_VERSION_MINOR" -ge 12 ]; then
    print_info "Python 3.12+ detected - updating pybind11 submodule for compatibility"
    print_info "Current pybind11 version is too old (2.6.0) - needs 2.11.1+ for Python 3.12"
    
    cd pybind11
    CURRENT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "")
    
    # Fetch latest tags from remote
    print_info "Fetching pybind11 tags..."
    git fetch origin --tags 2>/dev/null || git fetch --tags 2>/dev/null || true
    
    # Try to checkout a Python 3.12 compatible version (in order of preference)
    PYBIND11_UPDATED=false
    for version in "v2.13.0" "v2.12.0" "v2.11.1" "v2.11.0"; do
        if git checkout "$version" 2>/dev/null; then
            print_info "✓ pybind11 updated to $version (Python 3.12 compatible)"
            PYBIND11_UPDATED=true
            break
        fi
    done
    
    # If tags don't work, try updating to latest master
    if [ "$PYBIND11_UPDATED" = false ]; then
        print_info "Trying to update to latest master branch..."
        git fetch origin master 2>/dev/null || true
        if git checkout origin/master 2>/dev/null || git checkout master 2>/dev/null; then
            print_info "✓ pybind11 updated to latest master"
            PYBIND11_UPDATED=true
        fi
    fi
    
    if [ "$PYBIND11_UPDATED" = false ]; then
        print_warn "⚠ Could not update pybind11 - build may fail with Python 3.12"
        print_warn "  You may need to manually update the pybind11 submodule"
        print_warn "  Try: cd pybind11 && git fetch && git checkout v2.13.0"
    fi
    
    cd ..
else
    print_info "Python version $PYTHON_VERSION_MAJOR.$PYTHON_VERSION_MINOR is compatible with current pybind11"
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

# Install core dependencies first (without torch-tools to avoid visdom dependency issue)
print_info "Installing core dependencies..."
$PIP_CMD install svgwrite svgpathtools cssutils numba scikit-image

# Try to install visdom first (torch-tools depends on it)
# visdom has build issues with newer Python/setuptools versions
print_info "Attempting to install visdom (torch-tools dependency)..."
VISDOM_INSTALLED=false

# Ensure setuptools is properly installed and pkg_resources is available
$PIP_CMD install --upgrade 'setuptools>=40.0' 2>/dev/null || true

# Method 1: Try with --no-build-isolation (uses current environment)
if $PIP_CMD install --no-build-isolation visdom 2>/dev/null; then
    print_info "✓ visdom installed successfully"
    VISDOM_INSTALLED=true
else
    # Method 2: Try installing older setuptools temporarily
    print_info "Trying alternative visdom installation method..."
    OLD_SETUPTOOLS=$($PIP_CMD show setuptools 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "82.0.0")
    if $PIP_CMD install 'setuptools<70' 2>/dev/null && \
       $PIP_CMD install visdom 2>/dev/null; then
        # Restore setuptools
        $PIP_CMD install --upgrade "setuptools" 2>/dev/null || true
        print_info "✓ visdom installed successfully (with setuptools workaround)"
        VISDOM_INSTALLED=true
    else
        # Restore setuptools if we downgraded it
        $PIP_CMD install --upgrade setuptools 2>/dev/null || true
        print_warn "✗ visdom installation failed - will install torch-tools without it"
        print_warn "  Core functionality will work, but some visualization features may be unavailable"
        print_warn "  The painterly rendering scripts will work fine without visdom"
    fi
fi

# Install torch-tools (needed for LPIPS perceptual loss)
print_info "Installing torch-tools..."
if [ "$VISDOM_INSTALLED" = true ]; then
    # visdom is installed, can install torch-tools normally
    if $PIP_CMD install torch-tools; then
        print_info "✓ torch-tools installed successfully"
    else
        print_warn "torch-tools installation had issues, but continuing..."
    fi
else
    # Install torch-tools without dependencies, then install what we need manually
    print_warn "Installing torch-tools without visdom dependency..."
    if $PIP_CMD install --no-deps torch-tools; then
        # Install torch-tools dependencies manually (except visdom)
        $PIP_CMD install coloredlogs tqdm 2>/dev/null || true
        print_info "✓ torch-tools installed (without visdom)"
    else
        print_warn "torch-tools installation failed, but continuing..."
    fi
fi

# Install cmake if not available
if ! command -v cmake &> /dev/null; then
    print_info "Installing cmake via pip..."
    $PIP_CMD install cmake || print_warn "Failed to install cmake via pip. Please install cmake manually."
fi

echo ""

# Step 4: Verify dependencies
print_info "Verifying dependencies..."

# Check core dependencies (skip ttools/visdom since we handled them separately)
CORE_DEPS=("numpy" "skimage" "svgwrite" "svgpathtools" "cssutils" "numba")
MISSING_CORE=()

for dep in "${CORE_DEPS[@]}"; do
    # Handle skimage -> scikit-image import name
    import_name="$dep"
    if [ "$dep" = "skimage" ]; then
        import_name="skimage"
    fi
    
    if python3 -c "import $import_name" 2>/dev/null; then
        print_info "✓ $dep available"
    else
        MISSING_CORE+=("$dep")
    fi
done

# Check ttools (from torch-tools package) - don't try to install if already handled
if python3 -c "import ttools" 2>/dev/null; then
    print_info "✓ ttools module available"
elif $PIP_CMD show torch-tools >/dev/null 2>&1; then
    print_warn "torch-tools package installed but ttools import failed"
    print_warn "This may still work at runtime - continuing..."
else
    print_warn "torch-tools not found - but we should have installed it above"
fi

# Check visdom (optional)
if python3 -c "import visdom" 2>/dev/null; then
    print_info "✓ visdom available (optional)"
else
    print_warn "visdom not available (optional - only needed for some visualization scripts)"
fi

# Install any missing core dependencies
if [ ${#MISSING_CORE[@]} -gt 0 ]; then
    print_warn "Missing core dependencies: ${MISSING_CORE[*]}"
    print_info "Installing missing dependencies..."
    for dep in "${MISSING_CORE[@]}"; do
        case $dep in
            skimage)
                $PIP_CMD install scikit-image
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
