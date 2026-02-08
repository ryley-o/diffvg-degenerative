"""
Custom painterly rendering script for batch processing with strokes (GPU REQUIRED).

This script REQUIRES a CUDA-capable GPU and will fail if one is not available.
All tensors are explicitly placed on GPU for optimal performance.

This script always uses stroke format with 3 segments per path.
Outputs are saved to painterly-out/<input_filename>/ directory.
"""
import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import json
import os
import sys
from pathlib import Path

pydiffvg.set_print_timing(True)

gamma = 1.0
NUM_SEGMENTS = 3  # Always use 3 segments for strokes

def extract_path_data(shapes, shape_groups, canvas_width, canvas_height):
    """
    Extract path data from shapes and shape_groups for JSON export.
    
    Returns a list of path dictionaries with all information needed to rebuild strokes.
    """
    paths_data = []
    
    for i, (shape, group) in enumerate(zip(shapes, shape_groups)):
        # Extract points (convert from tensor to list, detach first)
        points = shape.points.detach().cpu().numpy().tolist()
        
        # Extract stroke color (RGBA, detach first)
        stroke_color = group.stroke_color.detach().cpu().numpy().tolist()
        
        # Extract stroke width (scalar, detach first)
        stroke_width = shape.stroke_width.detach().cpu().item()
        
        path_data = {
            "points": points,
            "stroke_color": stroke_color,
            "stroke_width": stroke_width
        }
        
        paths_data.append(path_data)
    
    return paths_data

def load_paths_strokes_json(input_path):
    """
    Load path data from JSON file with stroke format.
    
    Returns: (paths_data, canvas_width, canvas_height)
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if data.get('format') != 'stroke':
        raise ValueError(f"Input file format must be 'stroke', got '{data.get('format')}'")
    
    canvas_width = data['canvas']['width']
    canvas_height = data['canvas']['height']
    paths_data = data['paths']
    
    return paths_data, canvas_width, canvas_height

def save_paths_strokes_json(paths_data, canvas_width, canvas_height, output_path):
    """
    Save path data to JSON file with the specified format for strokes.
    """
    output_data = {
        "canvas": {
            "width": int(canvas_width),
            "height": int(canvas_height)
        },
        "num_segments": NUM_SEGMENTS,
        "format": "stroke",
        "paths": paths_data
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def main(args):
    # GPU REQUIRED - Check availability and fail early if not available
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is not available!")
        print("This script REQUIRES a CUDA-capable GPU.")
        print("Please run on a machine with GPU support or use the non-GPU version.")
        sys.exit(1)
    
    # Force GPU usage
    pydiffvg.set_use_gpu(True)
    device = pydiffvg.get_device()
    
    # Print GPU information
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 60)
    print("GPU REQUIRED MODE - CUDA GPU Detected")
    print("=" * 60)
    print(f"GPU Device: {gpu_name}")
    print(f"GPU Count: {gpu_count}")
    print(f"PyTorch Device: {device}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("=" * 60)
    print()
    
    # Move LPIPS model to GPU
    perception_loss = ttools.modules.LPIPS().to(device)
    
    # Determine if we're loading from existing paths or starting fresh
    load_from_paths = args.input_paths_file is not None
    
    if load_from_paths:
        # Load existing paths from JSON file
        print(f"Loading paths from: {args.input_paths_file}")
        paths_data, canvas_width, canvas_height = load_paths_strokes_json(args.input_paths_file)
        num_paths = len(paths_data)
        print(f"Loaded {num_paths} paths from existing file")
        print(f"Canvas dimensions: {canvas_width}x{canvas_height}")
        
        # Load target image (use same dimensions as paths file)
        # Try to find corresponding image in source_img directory
        input_basename = Path(args.input_paths_file).stem.replace('paths_strokes', '').replace('_iterplus', '').strip('_')
        if not input_basename:
            # Try to infer from directory structure
            path_parts = Path(args.input_paths_file).parts
            if 'painterly-out' in path_parts:
                idx = path_parts.index('painterly-out')
                if idx + 1 < len(path_parts):
                    input_basename = path_parts[idx + 1]
        
        # Try common image extensions
        input_image_path = None
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            candidate = os.path.join('source_img', input_basename + ext)
            if os.path.exists(candidate):
                input_image_path = candidate
                break
        
        if input_image_path is None:
            raise FileNotFoundError(f"Could not find source image for {input_basename}. Please specify --input_image")
        
        if args.input_image:
            input_image_path = args.input_image
    else:
        # Construct input image path
        input_image_path = os.path.join('source_img', args.input_filename)
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    # Load target image and move to GPU
    print(f"Loading target image: {input_image_path}")
    target = torch.from_numpy(skimage.io.imread(input_image_path)).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(device)  # Explicitly move to GPU
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW
    print(f"Target image moved to GPU: {target.device}")
    
    if not load_from_paths:
        canvas_width, canvas_height = target.shape[3], target.shape[2]
        num_paths = args.num_paths
    
    # Verify canvas dimensions match
    if target.shape[3] != canvas_width or target.shape[2] != canvas_height:
        raise ValueError(f"Target image dimensions ({target.shape[3]}x{target.shape[2]}) "
                        f"do not match paths canvas dimensions ({canvas_width}x{canvas_height})")
    
    # Set up output directory
    if load_from_paths:
        # Extract directory from input paths file
        input_path_obj = Path(args.input_paths_file)
        output_dir = input_path_obj.parent
        base_name = input_path_obj.stem
    else:
        input_basename = Path(args.input_filename).stem
        output_dir = os.path.join('painterly-out', input_basename)
        base_name = 'paths_strokes'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set stroke width constraints
    min_width = args.min_width if args.min_width is not None else 1.0
    max_width = args.max_width
    
    random.seed(1234)
    torch.manual_seed(1234)
    
    # Create shapes and shape groups - ALL tensors explicitly on GPU
    shapes = []
    shape_groups = []
    
    print(f"Creating shapes on GPU device: {device}")
    
    if load_from_paths:
        # Load shapes from existing paths data
        print("Initializing shapes from loaded paths...")
        for path_data in paths_data:
            # Create tensors directly on GPU
            points = torch.tensor(path_data['points'], dtype=torch.float32, device=device)
            stroke_width = torch.tensor(path_data['stroke_width'], dtype=torch.float32, device=device)
            stroke_color = torch.tensor(path_data['stroke_color'], dtype=torch.float32, device=device)
            
            # Clamp stroke width to new min/max if specified
            if args.min_width is not None or args.max_width is not None:
                stroke_width = torch.clamp(stroke_width, min_width, max_width)
            
            num_segments = NUM_SEGMENTS
            num_control_points = torch.zeros(num_segments, dtype=torch.int32, device=device) + 2
            
            path = pydiffvg.Path(num_control_points=num_control_points,
                                points=points,
                                stroke_width=stroke_width,
                                is_closed=False)
            shapes.append(path)
            
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1], device=device),
                                            fill_color=None,
                                            stroke_color=stroke_color)
            shape_groups.append(path_group)
    else:
        # Create new random shapes - ALL tensors on GPU
        print(f"Creating {num_paths} new random paths on GPU...")
        for i in range(num_paths):
            num_segments = NUM_SEGMENTS  # Always 3 segments
            num_control_points = torch.zeros(num_segments, dtype=torch.int32, device=device) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            # Create tensor directly on GPU
            points = torch.tensor(points, dtype=torch.float32, device=device)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 stroke_width=torch.tensor(1.0, device=device),
                                 is_closed=False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1], device=device),
                                             fill_color=None,
                                             stroke_color=torch.tensor([random.random(),
                                                                        random.random(),
                                                                        random.random(),
                                                                        random.random()], device=device))
            shape_groups.append(path_group)
    
    print(f"Created {len(shapes)} shapes, all on GPU device: {device}")
    
    # Set up optimization variables
    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    for path in shapes:
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)
    
    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)
    
    render = pydiffvg.RenderFunction.apply
    
    # Determine number of iterations to run
    if load_from_paths:
        # When loading from file, use additional_iterations for the actual iterations
        num_iterations = args.additional_iterations if args.additional_iterations > 0 else args.num_iter
    else:
        # When starting fresh, use num_iter
        num_iterations = args.num_iter
    
    print(f"\nStarting optimization with {num_iterations} iterations on GPU...")
    print()
    
    # Adam iterations
    for t in range(num_iterations):
        print('iteration:', t)
        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()
        
        # Forward pass: render the image
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     t,   # seed
                     None,
                     *scene_args)
        
        # Compose img with white background - explicitly use GPU device
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        
        if args.use_lpips_loss:
            loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
        else:
            loss = (img - target).pow(2).mean()
        print('render loss:', loss.item())
    
        # Backpropagate the gradients
        loss.backward()

        # Take a gradient descent step
        points_optim.step()
        width_optim.step()
        color_optim.step()
        
        # Clamp stroke widths to valid range
        for path in shapes:
            path.stroke_width.data.clamp_(min_width, max_width)
        
        # Clamp colors to valid range
        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)
    
    # Render the final result
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    
    # Compose final image with white background - explicitly use GPU device
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    
    # Determine output filenames
    if load_from_paths:
        # Append _iterplusxxx to base name
        iter_suffix = f"_iterplus{args.additional_iterations:03d}"
        # Remove any existing _iterplus suffix from base_name if present
        if '_iterplus' in base_name:
            base_name = base_name.rsplit('_iterplus', 1)[0]
        output_base = base_name + iter_suffix
    else:
        output_base = base_name
    
    final_svg_path = os.path.join(output_dir, output_base + '.svg')
    final_jpg_path = os.path.join(output_dir, output_base + '.jpg')
    final_json_path = os.path.join(output_dir, output_base + '.json')
    
    # Save SVG
    pydiffvg.save_svg(final_svg_path, canvas_width, canvas_height, shapes, shape_groups)
    
    # Save JPG (convert from GPU tensor to CPU, then save)
    img_cpu = img.cpu()
    img_np = (img_cpu * 255.0).clamp(0, 255).byte().numpy()
    skimage.io.imsave(final_jpg_path, img_np)
    
    # Extract and save path data as JSON
    paths_data = extract_path_data(shapes, shape_groups, canvas_width, canvas_height)
    save_paths_strokes_json(paths_data, canvas_width, canvas_height, final_json_path)
    
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - {output_base}.svg")
    print(f"  - {output_base}.jpg")
    print(f"  - {output_base}.json")
    print(f"\nGPU processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom painterly rendering with stroke format (3 segments) - GPU REQUIRED")
    
    # Input options - either input_filename OR input_paths_file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_filename", help="Input image filename (must be in source_img directory)")
    input_group.add_argument("--input_paths_file", help="Input paths JSON file to continue optimization from")
    
    parser.add_argument("--input_image", help="Source image path (required when using --input_paths_file if auto-detection fails)")
    parser.add_argument("--num_paths", type=int, default=512, help="Number of paths to generate (ignored when loading from file)")
    parser.add_argument("--min_width", type=float, default=None, help="Minimum stroke width (default: 1.0, or existing min when loading)")
    parser.add_argument("--max_width", type=float, default=2.0, help="Maximum stroke width")
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true', help="Use LPIPS perceptual loss")
    parser.add_argument("--num_iter", type=int, default=500, help="Number of optimization iterations (used when starting fresh, ignored when loading from file)")
    parser.add_argument("--additional_iterations", type=int, default=0, help="Number of additional iterations to run when loading from file (also used for output naming). If 0, uses --num_iter value.")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_paths_file:
        # When loading from file, additional_iterations is the primary parameter
        if args.additional_iterations == 0:
            # If not specified, use num_iter as fallback
            args.additional_iterations = args.num_iter
        if args.additional_iterations <= 0:
            raise ValueError("--additional_iterations must be > 0 when loading from file")
    
    main(args)
