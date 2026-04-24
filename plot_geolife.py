import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image
import urllib.request
import io
import pickle
import os
import sys

# Add the path to the traj module
sys.path.append('RLSTCcode/subtrajcluster')

# -------------------------------------------------------------
# 1. Slippy Map Math for OSM Tiles
# -------------------------------------------------------------
def deg2num(lat, lon, zoom):
    """Converts lat/lon to OpenStreetMap tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def num2deg(xtile, ytile, zoom):
    """Converts OpenStreetMap tile coordinates to lat/lon bounded corners."""
    n = 2.0 ** zoom
    lon = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat = math.degrees(lat_rad)
    return lat, lon

def add_osm_background(ax, zoom=11):
    """Downloads and tile-stitches OpenStreetMap images for the current ax boundaries."""
    lon_min, lon_max = ax.get_xlim()
    lat_min, lat_max = ax.get_ylim()
    
    xmin, ymax = deg2num(lat_min, lon_min, zoom)
    xmax, ymin = deg2num(lat_max, lon_max, zoom)
    
    # Ensure sane tile ranges
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)

    tiles = []
    total_tiles = (xmax - xmin + 1) * (ymax - ymin + 1)
    
    if total_tiles > 500:
        print(f"Warning: Attempting to download {total_tiles} tiles. Clamping zoom out.")
        return add_osm_background(ax, zoom=zoom-1)

    print(f"Fetching {total_tiles} map tiles from OpenStreetMap (zoom={zoom})...")
    
    for x in range(xmin, xmax + 1):
        column = []
        for y in range(ymin, ymax + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            req = urllib.request.Request(url, headers={'User-Agent': 'RLSTC-Viz/1.0'})
            try:
                with urllib.request.urlopen(req) as response:
                    img = Image.open(io.BytesIO(response.read())).convert("RGBA")
                    column.append(img)
            except Exception as e:
                # Fallback to an empty transparent tile if download fails
                column.append(Image.new('RGBA', (256, 256), (0, 0, 0, 0)))
        tiles.append(column)
    
    # Stitch column by column vertically
    col_imgs = []
    for col in tiles:
        widths, heights = zip(*(i.size for i in col))
        total_height = sum(heights)
        max_width = max(widths)
        col_img = Image.new('RGBA', (max_width, total_height))
        y_offset = 0
        for i in col:
            col_img.paste(i, (0, y_offset))
            y_offset += i.size[1]
        col_imgs.append(col_img)
        
    # Stitch horizontally
    widths, heights = zip(*(i.size for i in col_imgs))
    total_width = sum(widths)
    max_height = max(heights)
    stitched = Image.new('RGBA', (total_width, max_height))
    x_offset = 0
    for i in col_imgs:
        stitched.paste(i, (x_offset, 0))
        x_offset += i.size[0]
        
    # Get geographic exact edges of the stitched image matrix
    lat_top, lon_left = num2deg(xmin, ymin, zoom)
    lat_bottom, lon_right = num2deg(xmax+1, ymax+1, zoom)
    
    # Render behind (zorder=-10) and slightly transparent
    ax.imshow(stitched, extent=(lon_left, lon_right, lat_bottom, lat_top), 
              aspect='auto', alpha=0.6, zorder=-10)


# -------------------------------------------------------------
# 2. Extract Trajectories & Plot
# -------------------------------------------------------------
def plot_trajectories():
    fig, ax = plt.subplots(figsize=(10, 10))
    has_plotted_anything = False
    all_x = []
    all_y = []
    
    # --- BLUE LINES: 1000 Sampled Sub-trajectories ---
    subtraj_paths = ["data/geo-subtrajs", "RLSTCcode/data/ied_subtrajs1000", "data/Tdrive_testdata", "data/geolife_testdata"]
    loaded_blues = False
    for path in subtraj_paths:
        if os.path.exists(path):
            import random
            print(f"Loading background sub-trajectories from: {path}")
            with open(path, 'rb') as f:
                subtrajs = pickle.load(f)
            
            lines_blue = []
            sample_size = min(1000, len(subtrajs))
            sampled = random.sample(subtrajs, sample_size)
            
            for t in sampled:
                if hasattr(t, 'points'):
                    pts = [(p.x, p.y) for p in t.points] # Coordinates for LineCollection
                elif isinstance(t, list):
                    pts = t
                else:
                    continue
                if len(pts) > 1:
                    lines_blue.append(pts)
                    
            if lines_blue:
                lc = LineCollection(lines_blue, colors="blue", linewidths=0.3, alpha=0.4)
                ax.add_collection(lc)
                all_x.extend([p[0] for line in lines_blue for p in line])
                all_y.extend([p[1] for line in lines_blue for p in line])
                has_plotted_anything = True
                loaded_blues = True
            break
            
    if not loaded_blues:
        print("Warning: Could not find pickled sub-trajectories (blue lines).")

    # --- RED LINES: Cluster Reps ---
    cluster_paths = ["RLSTCcode/data/geolife_rlstc_clusters", "RLSTCcode/data/tdrive_clustercenter_10", "RLSTCcode/data/geolife_clustercenter", "RLSTCcode/data/tdrive_clustercenter"]
    loaded_reds = False
    for cp in cluster_paths:
        if os.path.exists(cp):
            print(f"Loading cluster representative lines from: {cp}")
            with open(cp, 'rb') as f:
                result = pickle.load(f)
            
            lines_red = []
            # Format parsing (Supports the advanced structure from the documentation block)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) and len(result[0]) > 2:
                for c in result[0][2].values():
                    # Extracts the "Classical segments" from index [3][0] of each cluster
                    if len(c) > 3 and len(c[3]) > 0 and hasattr(c[3][0], 'points'):
                        pts = [(p.x, p.y) for p in c[3][0].points]
                        if len(pts) > 1:
                            lines_red.append(pts)
            # Alternative simpler format fallback
            elif isinstance(result, list):
                for t in result:
                    if hasattr(t, 'points'):
                        pts = [(p.x, p.y) for p in t.points]
                        if len(pts) > 1:
                            lines_red.append(pts)
                            
            if lines_red:
                lc_red = LineCollection(lines_red, colors="red", linewidths=1.5, alpha=0.9, zorder=5)
                ax.add_collection(lc_red)
                
                # Update bounds prioritizing red centroids
                if not all_x:
                    all_x.extend([p[0] for line in lines_red for p in line])
                    all_y.extend([p[1] for line in lines_red for p in line])
                
                has_plotted_anything = True
                loaded_reds = True
            break

    if not loaded_reds:
        print("Warning: Could not find pickled cluster centroids (red lines).")
            
    # Set window to exact spatial bounds of the data loaded
    if has_plotted_anything and all_x and all_y:
        ax.set_xlim(min(all_x), max(all_x))
        ax.set_ylim(min(all_y), max(all_y))
    else:
        # Fallback arbitrary Beijing bounds
        ax.set_xlim(116.1, 116.7)
        ax.set_ylim(39.7, 40.2)
        print("Warning: No paths found to draw. Rendering blank map of Beijing.")
        
    ax.set_title("Recreated Fig. 16: Geolife (Classical DQN)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # 3. Add Map Background exactly mimicking OSM logic
    add_osm_background(ax, zoom=11)
    
    # Output to File
    out_img = "recreated_classical_geolife_fig16.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"✅ Success: Rendered geometry to {out_img}")

if __name__ == "__main__":
    plot_trajectories()
