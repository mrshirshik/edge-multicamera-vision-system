"""
Multi-Object Tracking Module
DeepSORT-based tracking implementation
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Single track in MOT"""
    track_id: int
    detections: List[Dict] = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False


class MultiObjectTracker:
    """Multi-Object Tracker using DeepSORT approach"""
    
    def __init__(self, max_age: int = 70, min_hits: int = 3):
        """
        Initialize tracker
        
        Args:
            max_age: Maximum frames to keep unconfirmed track
            min_hits: Minimum detections to confirm track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}
        self.frame_count = 0
    
    def update(self, detections: List[Dict], frame_idx: Optional[int] = None) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections with bbox, class, conf
            frame_idx: Frame index
            
        Returns:
            List of active tracks
        """
        self.frame_count = frame_idx if frame_idx is not None else self.frame_count + 1
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_trks = self._match_detections(detections)
        
        # Update matched tracks
        for trk_idx, det_idx in matched:
            track = list(self.tracks.values())[trk_idx]
            track.detections.append(detections[det_idx])
            track.bbox_history.append(detections[det_idx]['bbox'])
            track.hits += 1
            track.misses = 0
            track.last_frame = self.frame_count
            
            if track.hits >= self.min_hits:
                track.confirmed = True
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                track_id=self.next_track_id,
                detections=[detections[det_idx]],
                first_frame=self.frame_count,
                last_frame=self.frame_count
            )
            new_track.bbox_history.append(detections[det_idx]['bbox'])
            
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
        
        # Handle unmatched tracks
        for trk_idx in unmatched_trks:
            track = list(self.tracks.values())[trk_idx]
            track.misses += 1
            track.age += 1
        
        # Remove dead tracks
        self._prune_tracks()
        
        return self.get_confirmed_tracks()
    
    def _match_detections(self, detections: List[Dict]) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Match detections to tracks using IoU
        
        Args:
            detections: New detections
            
        Returns:
            (matched_pairs, unmatched_det_indices, unmatched_trk_indices)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(detections)
        
        # Simple greedy matching
        matched = []
        matched_trks = set()
        matched_dets = set()
        
        for trk_idx in range(len(self.tracks)):
            best_det_idx = -1
            best_iou = 0.3  # Minimum IoU threshold
            
            for det_idx in range(len(detections)):
                if det_idx in matched_dets:
                    continue
                
                iou = iou_matrix[trk_idx, det_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                matched.append((trk_idx, best_det_idx))
                matched_trks.add(trk_idx)
                matched_dets.add(best_det_idx)
        
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_trks = [i for i in range(len(self.tracks)) if i not in matched_trks]
        
        return matched, unmatched_dets, unmatched_trks
    
    def _compute_iou_matrix(self, detections: List[Dict]) -> np.ndarray:
        """Compute IoU matrix between tracks and detections"""
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        iou_matrix = np.zeros((num_tracks, num_dets))
        
        tracks_list = list(self.tracks.values())
        
        for i, track in enumerate(tracks_list):
            if not track.bbox_history:
                continue
            
            last_bbox = track.bbox_history[-1]
            
            for j, det in enumerate(detections):
                det_bbox = det['bbox']
                iou = self._bbox_iou(last_bbox, det_bbox)
                iou_matrix[i, j] = iou
        
        return iou_matrix
    
    @staticmethod
    def _bbox_iou(bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute IoU between two bboxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _prune_tracks(self) -> None:
        """Remove dead tracks"""
        to_remove = []
        
        for track_id, track in self.tracks.items():
            if track.misses > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get all confirmed tracks"""
        return [t for t in self.tracks.values() if t.confirmed]
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks"""
        return list(self.tracks.values())
    
    def reset(self) -> None:
        """Reset tracker"""
        self.tracks.clear()
        self.frame_count = 0
        self.next_track_id = 1
