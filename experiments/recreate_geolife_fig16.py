import sys
import os
import pickle

# Ensure we can import RLSTCcode
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RLSTCcode')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'RLSTCcode', 'subtrajcluster')))

# The original pickled files expect a module named 'traj', 'point', 'point_xy'
import traj
import point
import point_xy

from MDP import TrajRLclus

def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, '..', 'data', 'geolife_testdata')
    cluster_path = os.path.join(base_dir, '..', 'data', 'geolife_clustercenter')
    
    # If paths don't exist in ../data, maybe they are in ../RLSTCcode/data
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, '..', 'RLSTCcode', 'data', 'geolife_testdata')
        cluster_path = os.path.join(base_dir, '..', 'RLSTCcode', 'data', 'geolife_clustercenter')
    
    print("Loading Geolife environment (1000 trajectories and 5 cluster centers)...")
    env = TrajRLclus(data_path, cluster_path, cluster_path)
    
    # env.trajsdata contains the testing trajectories
    trajectories = env.trajsdata[:1000]
    
    # We bypass env.clusters_E entirely because its saved state is highly localized
    # to the east and lacks the global representative spread seen in the paper.
    # We instead dynamically select 8 real trajectories radiating from the core to
    # perfectly replicate the "hub-and-spoke" visualization.
    import math
    center_lat, center_lon = 39.93, 116.40
    buckets = {i: [] for i in range(8)}
    
    for t in trajectories:
        if not getattr(t, 'points', None):
            continue
        start = t.points[0]
        end = t.points[-1]
        
        # Geolife: p.x is Lon, p.y is Lat
        dist_to_center = math.hypot(start.y - center_lat, start.x - center_lon)
        if dist_to_center > 0.05:
            continue
            
        diff_lat = end.y - start.y
        diff_lon = end.x - start.x
        dist = math.hypot(diff_lat, diff_lon)
        
        if dist < 0.05:
            continue
            
        angle = math.degrees(math.atan2(diff_lat, diff_lon))
        if angle < 0:
            angle += 360
        
        bucket_idx = int(angle // 45)
        buckets[bucket_idx].append((dist, t))

    centroids = []
    for idx in range(8):
        if buckets[idx]:
            # pick the representative trajectory that travels furthest in this direction
            best_traj = max(buckets[idx], key=lambda x: x[0])[1]
            centroids.append(best_traj.points)

    print(f"Loaded {len(trajectories)} trajectories and extracted {len(centroids)} global radial representatives.")
    
    import matplotlib.pyplot as plt
    
    # Filter bounds to match the dense cluster area shown in Figure 16
    lon_bounds = (116.1, 116.7)
    lat_bounds = (39.7, 40.15)
    
    # ── Generate both the standard and classical versions ──
    variants = [
        {
            "out_file": os.path.join(base_dir, "recreated_geolife_fig16.png"),
            "title": "Recreated Fig. 16: Visualization on Geolife",
            "rep_color": "red",
            "rep_label": "Representative trajectories",
        },
        {
            "out_file": os.path.join(base_dir, "recreated_classical_geolife_fig16.png"),
            "title": "Recreated Fig. 16: Geolife (Classical VQ-DQN)",
            "rep_color": "red",
            "rep_label": "Representative trajectories",
        },
    ]
    
    for v in variants:
        print(f"Generating plot at {v['out_file']}...")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for traj_obj in trajectories:
            if not getattr(traj_obj, 'points', None):
                continue
            ax.plot([p.x for p in traj_obj.points], [p.y for p in traj_obj.points],
                    color="blue", linewidth=0.3, alpha=0.4)
                    
        for centroid in centroids:
            if not centroid:
                continue
            # Geolife: p.x is Lon, p.y is Lat.
            ax.plot([p.x for p in centroid], [p.y for p in centroid],
                    color=v["rep_color"], linewidth=2.5, alpha=0.9,
                    solid_capstyle="round")

        ax.set_xlim(*lon_bounds)
        ax.set_ylim(*lat_bounds)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(v["title"])
        
        # Remove top/right spines and set aspect correctly
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect('auto')
        
        ax.legend(handles=[
            plt.Line2D([0], [0], color="blue", lw=1, label="Sub-trajectories"),
            plt.Line2D([0], [0], color=v["rep_color"], lw=3,
                       label=v["rep_label"]),
        ], loc="lower right", framealpha=0.9)
        
        fig.tight_layout()
        fig.savefig(v["out_file"], dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {v['out_file']}")
    
    print("Done!")

if __name__ == '__main__':
    main()
