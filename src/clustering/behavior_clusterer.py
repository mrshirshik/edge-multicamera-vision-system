"""
Behavior Clustering Module
Clusters objects based on movement patterns and attributes
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class BehaviorCluster:
    """Behavior cluster"""
    cluster_id: int
    members: List[int]  # Track IDs
    centroid: np.ndarray
    characteristics: Dict[str, Any]


class BehaviorClusterer:
    """Clusters tracks based on behavior patterns"""
    
    def __init__(self, n_clusters: int = 3, method: str = 'kmeans'):
        """
        Initialize clusterer
        
        Args:
            n_clusters: Number of clusters
            method: 'kmeans' or 'dbscan'
        """
        self.n_clusters = n_clusters
        self.method = method
        self.scaler = StandardScaler()
        self.clusters: List[BehaviorCluster] = []
    
    def cluster_tracks(self, tracks: List[Dict]) -> List[BehaviorCluster]:
        """
        Cluster tracks by behavior
        
        Args:
            tracks: List of track data
            
        Returns:
            List of behavior clusters
        """
        if not tracks:
            return []
        
        try:
            # Extract features
            features = self._extract_features(tracks)
            
            if features.shape[0] < 2:
                return []
            
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Perform clustering
            if self.method == 'kmeans':
                labels = self._kmeans_cluster(features_scaled)
            else:
                labels = self._dbscan_cluster(features_scaled)
            
            # Create clusters
            self.clusters = self._create_clusters(tracks, labels, features)
            
            return self.clusters
            
        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return []
    
    def _extract_features(self, tracks: List[Dict]) -> np.ndarray:
        """Extract features from tracks"""
        features = []
        
        for track in tracks:
            bbox_history = track.get('bbox_history', [])
            
            if len(bbox_history) < 2:
                features.append([0, 0, 0, 0])
                continue
            
            # Extract velocity
            x1, y1, w1, h1 = bbox_history[-2]
            x2, y2, w2, h2 = bbox_history[-1]
            
            vx = x2 - x1
            vy = y2 - y1
            v_mag = np.sqrt(vx**2 + vy**2)
            
            # Extract size change
            size_change = (w2 * h2) - (w1 * h1)
            
            features.append([vx, vy, v_mag, size_change])
        
        return np.array(features, dtype=np.float32)
    
    def _kmeans_cluster(self, features: np.ndarray) -> np.ndarray:
        """K-Means clustering"""
        kmeans = KMeans(n_clusters=min(self.n_clusters, len(features)), 
                       random_state=42, n_init=10)
        return kmeans.fit_predict(features)
    
    def _dbscan_cluster(self, features: np.ndarray) -> np.ndarray:
        """DBSCAN clustering"""
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        return dbscan.fit_predict(features)
    
    def _create_clusters(self, tracks: List[Dict], labels: np.ndarray, 
                        features: np.ndarray) -> List[BehaviorCluster]:
        """Create cluster objects"""
        clusters = []
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Noise points in DBSCAN
                continue
            
            mask = labels == cluster_id
            member_indices = np.where(mask)[0]
            
            cluster = BehaviorCluster(
                cluster_id=int(cluster_id),
                members=[int(i) for i in member_indices],
                centroid=features[mask].mean(axis=0),
                characteristics=self._analyze_cluster(tracks, member_indices)
            )
            clusters.append(cluster)
        
        return clusters
    
    def _analyze_cluster(self, tracks: List[Dict], indices: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        cluster_tracks = [tracks[i] for i in indices]
        
        velocities = []
        for track in cluster_tracks:
            bbox_history = track.get('bbox_history', [])
            if len(bbox_history) >= 2:
                x1, y1, _, _ = bbox_history[-2]
                x2, y2, _, _ = bbox_history[-1]
                velocities.append(np.sqrt((x2-x1)**2 + (y2-y1)**2))
        
        return {
            'num_members': len(cluster_tracks),
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'max_velocity': np.max(velocities) if velocities else 0,
            'behavior_type': self._classify_behavior(velocities)
        }
    
    def _classify_behavior(self, velocities: List[float]) -> str:
        """Classify behavior based on velocity"""
        if not velocities:
            return 'static'
        
        avg_vel = np.mean(velocities)
        
        if avg_vel < 1:
            return 'static'
        elif avg_vel < 5:
            return 'slow_moving'
        elif avg_vel < 15:
            return 'medium_speed'
        else:
            return 'fast_moving'
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get clustering statistics"""
        return {
            'num_clusters': len(self.clusters),
            'total_members': sum(len(c.members) for c in self.clusters),
            'clusters': [
                {
                    'id': c.cluster_id,
                    'members': len(c.members),
                    'behavior': c.characteristics.get('behavior_type', 'unknown')
                }
                for c in self.clusters
            ]
        }
