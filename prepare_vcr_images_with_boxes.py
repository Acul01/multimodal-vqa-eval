#!/usr/bin/env python3
"""
prepare_vcr_images_with_boxes.py

Script to process all VCR images, draw bounding boxes from JSON annotations,
and save them in a new directory structure.

Usage:
    python prepare_vcr_images_with_boxes.py --images_root /path/to/images --output_dir vcr_images_with_boxes
    python prepare_vcr_images_with_boxes.py --images_root /path/to/images --output_dir vcr_images_with_boxes --draw_seg
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable


def find_json_for_image(image_path: str) -> str | None:
    """Find the corresponding JSON annotation file for an image."""
    base = os.path.splitext(image_path)[0]
    json_path = base + ".json"
    if os.path.exists(json_path):
        return json_path
    return None


def draw_annotations(
    image_path: str,
    json_path: str,
    output_path: str,
    draw_seg: bool = False,
    draw_labels: bool = True,
    line_width: int = 3,
) -> bool:
    """
    Draw bounding boxes from VCR JSON onto image and save to output_path.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load JSON {json_path}: {e}")
        return False
    
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return False
    
    draw = ImageDraw.Draw(img)
    
    # Load font
    font_size = max(24, img.width // 32)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except Exception:
        try:
            # Try common system fonts
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=font_size)
        except Exception:
            font = ImageFont.load_default()
    
    boxes = data.get('boxes', [])
    segms = data.get('segms', [])
    
    # Draw each box with its index (0, 1, 2, etc. corresponding to det0, det1, det2)
    for i, box in enumerate(boxes):
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        color = (255, 0, 0)  # Red boxes
        
        # Draw box outline
        for w in range(line_width):
            draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=color)
        
        if draw_labels:
            text = str(i)  # Shows "0", "1", "2" etc. (corresponds to det0, det1, det2)
            bbox = font.getbbox(text)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx1, ty1 = x1, max(0, y1 - text_h - 4)
            tx2, ty2 = tx1 + text_w + 4, ty1 + text_h + 4
            # Draw label background
            draw.rectangle([tx1, ty1, tx2, ty2], fill=(255, 0, 0))
            draw.text((tx1 + 2, ty1 + 2), text, fill=(255, 255, 255), font=font)
    
    # Optional: draw segmentation polygons
    if draw_seg and segms:
        seg_color = (0, 255, 0)  # Green polygons
        for obj in segms:
            if not obj:
                continue
            polygons = obj if isinstance(obj[0][0], (int, float)) else obj
            # Handle nested polygon structure
            if isinstance(polygons[0][0], (list, tuple)):
                for poly in polygons:
                    pts = [(int(x), int(y)) for x, y in poly]
                    draw.polygon(pts, outline=seg_color)
            else:
                # Single polygon as flat list of pairs
                pts = [(int(x), int(y)) for x, y in polygons]
                draw.polygon(pts, outline=seg_color)
    
    # Save annotated image
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save {output_path}: {e}")
        return False


def process_vcr_images(
    images_root: str,
    output_dir: str,
    draw_seg: bool = False,
    draw_labels: bool = True,
    line_width: int = 3,
):
    """
    Process all VCR images: find JSON annotations, draw boxes, save to output_dir.
    
    Args:
        images_root: Root directory containing vcr1images/ folder
        output_dir: Directory to save annotated images (will create vcr1images/ subdirectory)
        draw_seg: Whether to draw segmentation polygons
        draw_labels: Whether to draw box labels (indices)
        line_width: Width of bounding box lines
    """
    vcr_images_dir = os.path.join(images_root, "vcr1images")
    
    if not os.path.exists(vcr_images_dir):
        raise FileNotFoundError(f"VCR images directory not found: {vcr_images_dir}")
    
    # Create output directory structure
    output_vcr_dir = os.path.join(output_dir, "vcr1images")
    os.makedirs(output_vcr_dir, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    print(f"Scanning for images in {vcr_images_dir}...")
    for root, dirs, files in os.walk(vcr_images_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    processed = 0
    skipped_no_json = 0
    errors = 0
    
    for image_path in tqdm(image_files, desc="Processing images"):
        # Find corresponding JSON
        json_path = find_json_for_image(image_path)
        
        if json_path is None:
            skipped_no_json += 1
            # Copy original image if no JSON found
            rel_path = os.path.relpath(image_path, vcr_images_dir)
            output_path = os.path.join(output_vcr_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            try:
                import shutil
                shutil.copy2(image_path, output_path)
            except Exception as e:
                print(f"[WARN] Failed to copy {image_path}: {e}")
            continue
        
        # Determine output path (preserve directory structure)
        rel_path = os.path.relpath(image_path, vcr_images_dir)
        output_path = os.path.join(output_vcr_dir, rel_path)
        
        # Draw annotations
        success = draw_annotations(
            image_path,
            json_path,
            output_path,
            draw_seg=draw_seg,
            draw_labels=draw_labels,
            line_width=line_width,
        )
        
        if success:
            processed += 1
        else:
            errors += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary:")
    print(f"  Total images found: {len(image_files)}")
    print(f"  Successfully processed (with boxes): {processed}")
    print(f"  Skipped (no JSON found): {skipped_no_json}")
    print(f"  Errors: {errors}")
    print(f"  Output directory: {output_vcr_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Process all VCR images: draw bounding boxes from JSON annotations and save to new directory.'
    )
    parser.add_argument(
        '--images_root',
        type=str,
        required=True,
        help='Root directory containing vcr1images/ folder'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='vcr_images_with_boxes',
        help='Output directory for annotated images (default: vcr_images_with_boxes)'
    )
    parser.add_argument(
        '--draw_seg',
        action='store_true',
        help='Also draw segmentation polygons if present in JSON'
    )
    parser.add_argument(
        '--no_labels',
        dest='draw_labels',
        action='store_false',
        help='Do not draw box labels (indices)'
    )
    parser.add_argument(
        '--line_width',
        type=int,
        default=3,
        help='Bounding box line width (default: 3)'
    )
    
    args = parser.parse_args()
    
    process_vcr_images(
        images_root=args.images_root,
        output_dir=args.output_dir,
        draw_seg=args.draw_seg,
        draw_labels=args.draw_labels,
        line_width=args.line_width,
    )


if __name__ == '__main__':
    main()

