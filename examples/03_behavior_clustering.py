"""
Example 3: Behavior Clustering and Analysis
Cluster objects by movement behavior
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.clustering import BehaviorClusterer
from src.utils import setup_logger

logger = setup_logger("example_clustering", log_file="logs/example_clustering.log")


def main():
    """Run clustering example"""
    
    # Create clusterer
    clusterer = BehaviorClusterer(n_clusters=3, method='kmeans')
    
    # Create mock track data
    tracks = []
    
    # Static objects
    for i in range(5):
        tracks.append({
            'track_id': i,
            'bbox_history': [
                (100 + i*50, 100, 50, 50),
                (100 + i*50, 100, 50, 50),
                (100 + i*50, 100, 50, 50)
            ]
        })
    
    # Slow moving objects
    for i in range(5, 10):
        tracks.append({
            'track_id': i,
            'bbox_history': [
                (100, 200 + i*30, 40, 40),
                (102, 200 + i*30 + 2, 40, 40),
                (104, 200 + i*30 + 4, 40, 40)
            ]
        })
    
    # Fast moving objects
    for i in range(10, 15):
        tracks.append({
            'track_id': i,
            'bbox_history': [
                (200, 300 + i*40, 60, 60),
                (210, 300 + i*40 + 20, 60, 60),
                (220, 300 + i*40 + 40, 60, 60)
            ]
        })
    
    logger.info(f"Created {len(tracks)} mock tracks")
    
    # Cluster
    clusters = clusterer.cluster_tracks(tracks)
    
    logger.info(f"Generated {len(clusters)} clusters")
    
    for cluster in clusters:
        logger.info(f"Cluster {cluster.cluster_id}:")
        logger.info(f"  Members: {cluster.members}")
        logger.info(f"  Behavior: {cluster.characteristics.get('behavior_type')}")
        logger.info(f"  Avg Velocity: {cluster.characteristics.get('avg_velocity'):.2f}")
    
    # Get statistics
    stats = clusterer.get_cluster_stats()
    logger.info(f"\nCluster Statistics: {stats}")


if __name__ == "__main__":
    main()
