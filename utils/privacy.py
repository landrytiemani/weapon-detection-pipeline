"""
Privacy Protection Module for Weapon Detection Pipeline

Provides privacy-preserving features:
- Face blurring (pixelate or gaussian)
- Silhouette masking
- Selective application based on weapon detection results

Usage:
    from utils.privacy import PrivacyProtector
    
    privacy = PrivacyProtector(config['privacy'])
    frame = privacy.apply_privacy(frame, person_data, weapon_stats)
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class PrivacyProtector:
    """
    Applies privacy protection to individuals in frames based on weapon detection status.
    
    Supports two scopes:
    - "non_targets": Only blur people WITHOUT weapons
    - "everyone": Blur all detected people
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize privacy protector with configuration.
        
        Args:
            config: Privacy configuration dict containing:
                - enabled: bool, whether privacy is active
                - scope: str, "non_targets" or "everyone"
                - face_blur: dict with blur settings
                - silhouette: dict with masking settings
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        self.scope = config.get('scope', 'non_targets')
        
        # Face blur config
        self.face_blur_enabled = config.get('face_blur', {}).get('enabled', False)
        self.face_blur_method = config.get('face_blur', {}).get('method', 'pixelate')
        self.pixel_block_size = config.get('face_blur', {}).get('pixel_block', 15)
        self.gaussian_ksize = config.get('face_blur', {}).get('gaussian_ksize', 31)
        
        # Silhouette config
        self.silhouette_enabled = config.get('silhouette', {}).get('enabled', False)
        self.fill_color = tuple(config.get('silhouette', {}).get('fill_color', [32, 32, 32]))
        self.fill_alpha = config.get('silhouette', {}).get('fill_alpha', 0.88)
        self.outline_px = config.get('silhouette', {}).get('outline_px', 2)
        
        # Face detector (if needed)
        self.face_detector = None
        if self.face_blur_enabled:
            detector_type = config.get('face_blur', {}).get('detector', 'none')
            if detector_type == 'haar':
                self._init_haar_detector(config.get('face_blur', {}).get('haar_path'))
            elif detector_type == 'yunet':
                self._init_yunet_detector()
            # else: detector_type == 'none', use bounding box estimation
        
        if self.enabled:
            print(f"[PRIVACY] Enabled - Scope: {self.scope}")
            if self.face_blur_enabled:
                print(f"[PRIVACY]   Face blur: {self.face_blur_method}")
            if self.silhouette_enabled:
                print(f"[PRIVACY]   Silhouette masking: enabled")
    
    
    def _init_haar_detector(self, haar_path: Optional[str] = None):
        """Initialize Haar Cascade face detector"""
        try:
            if not haar_path:
                haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(haar_path)
            if self.face_detector.empty():
                print("[WARN] Failed to load Haar cascade, falling back to bbox estimation")
                self.face_detector = None
            else:
                print(f"[PRIVACY] Loaded Haar cascade from: {haar_path}")
        except Exception as e:
            print(f"[WARN] Could not load Haar detector: {e}")
            self.face_detector = None
    
    
    def _init_yunet_detector(self):
        """Initialize YuNet face detector (OpenCV DNN)"""
        try:
            # YuNet requires model file - if not available, fall back to bbox estimation
            print("[WARN] YuNet detector not implemented, using bbox estimation")
            self.face_detector = None
        except Exception as e:
            print(f"[WARN] Could not load YuNet detector: {e}")
            self.face_detector = None
    
    
    def apply_privacy(self, 
                     frame: np.ndarray, 
                     person_data: List[Dict[str, Any]], 
                     weapon_stats: Dict[str, Any]) -> np.ndarray:
        """
        Apply privacy protection to frame based on weapon detection results.
        
        Args:
            frame: Input frame (BGR)
            person_data: List of person detections with 'id' and 'bbox'
            weapon_stats: Weapon detection results containing 'weapon_person_ids'
        
        Returns:
            Frame with privacy protection applied
        """
        if not self.enabled or len(person_data) == 0:
            return frame
        
        # Get IDs of people with weapons
        weapon_person_ids = set(weapon_stats.get('weapon_person_ids', []))
        
        # Create a copy to avoid modifying original
        protected_frame = frame.copy()
        
        for person in person_data:
            person_id = person.get('id')
            bbox = person.get('bbox')
            
            if not bbox or len(bbox) < 4:
                continue
            
            # Determine if this person should have privacy applied
            should_protect = self._should_apply_privacy(person_id, weapon_person_ids)
            
            if should_protect:
                # Apply face blur if enabled
                if self.face_blur_enabled:
                    protected_frame = self._apply_face_blur(protected_frame, bbox)
                
                # Apply silhouette if enabled
                if self.silhouette_enabled:
                    protected_frame = self._apply_silhouette(protected_frame, bbox)
        
        return protected_frame
    
    
    def _should_apply_privacy(self, person_id: int, weapon_person_ids: set) -> bool:
        """
        Determine if privacy should be applied to a person.
        
        Args:
            person_id: ID of the person
            weapon_person_ids: Set of person IDs with weapons
        
        Returns:
            True if privacy should be applied, False otherwise
        """
        if self.scope == "everyone":
            return True
        elif self.scope == "non_targets":
            # Apply privacy only to people WITHOUT weapons
            return person_id not in weapon_person_ids
        return False
    
    
    def _apply_face_blur(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Apply face blurring within person bounding box.
        
        Args:
            frame: Input frame
            bbox: Person bounding box [x1, y1, x2, y2]
        
        Returns:
            Frame with face blurred
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure valid coordinates
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return frame
        
        # Extract person region
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return frame
        
        # Detect face region
        face_bbox = self._detect_face_region(person_roi, x1, y1)
        
        if face_bbox is None:
            return frame
        
        fx1, fy1, fx2, fy2 = face_bbox
        
        # Extract face ROI
        face_roi = frame[fy1:fy2, fx1:fx2]
        
        if face_roi.size == 0:
            return frame
        
        # Apply blur based on method
        if self.face_blur_method == 'pixelate':
            blurred_face = self._pixelate(face_roi, self.pixel_block_size)
        else:  # gaussian
            blurred_face = self._gaussian_blur(face_roi, self.gaussian_ksize)
        
        # Replace face region with blurred version
        frame[fy1:fy2, fx1:fx2] = blurred_face
        
        return frame
    
    
    def _detect_face_region(self, 
                           person_roi: np.ndarray, 
                           offset_x: int, 
                           offset_y: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face region within person ROI.
        
        Args:
            person_roi: Person region of interest
            offset_x: X offset of person ROI in full frame
            offset_y: Y offset of person ROI in full frame
        
        Returns:
            Face bounding box [x1, y1, x2, y2] in full frame coordinates, or None
        """
        roi_h, roi_w = person_roi.shape[:2]
        
        # Method 1: Use Haar cascade if available
        if self.face_detector is not None:
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Take the largest detected face
                areas = [w * h for (x, y, w, h) in faces]
                largest_idx = np.argmax(areas)
                fx, fy, fw, fh = faces[largest_idx]
                
                # Convert to full frame coordinates
                fx1 = offset_x + fx
                fy1 = offset_y + fy
                fx2 = fx1 + fw
                fy2 = fy1 + fh
                
                return (fx1, fy1, fx2, fy2)
        
        # Method 2: Fallback to heuristic estimation
        # Assume face is in upper 35-40% of person bbox
        face_height = int(roi_h * 0.38)
        face_y1 = offset_y + int(roi_h * 0.05)  # Small offset from top
        face_y2 = face_y1 + face_height
        
        # Face width is typically centered and narrower than body
        face_width = int(roi_w * 0.7)
        face_x1 = offset_x + int((roi_w - face_width) / 2)
        face_x2 = face_x1 + face_width
        
        return (face_x1, face_y1, face_x2, face_y2)
    
    
    def _pixelate(self, roi: np.ndarray, block_size: int = 15) -> np.ndarray:
        """
        Apply pixelation effect to region.
        
        Args:
            roi: Region of interest to pixelate
            block_size: Size of pixelation blocks
        
        Returns:
            Pixelated region
        """
        h, w = roi.shape[:2]
        
        if h < block_size or w < block_size:
            return roi
        
        # Downsample
        temp_h = max(1, h // block_size)
        temp_w = max(1, w // block_size)
        temp = cv2.resize(roi, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR)
        
        # Upsample with nearest neighbor to create pixelated effect
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    
    def _gaussian_blur(self, roi: np.ndarray, ksize: int = 31) -> np.ndarray:
        """
        Apply Gaussian blur to region.
        
        Args:
            roi: Region of interest to blur
            ksize: Kernel size (must be odd)
        
        Returns:
            Blurred region
        """
        # Ensure kernel size is odd
        if ksize % 2 == 0:
            ksize += 1
        
        # Ensure minimum size
        ksize = max(3, ksize)
        
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        return blurred
    
    
    def _apply_silhouette(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Apply silhouette masking to person bounding box.
        
        Args:
            frame: Input frame
            bbox: Person bounding box [x1, y1, x2, y2]
        
        Returns:
            Frame with silhouette applied
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure valid coordinates
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return frame
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.fill_color, -1)
        
        # Blend overlay with original frame
        frame = cv2.addWeighted(
            frame, 
            1 - self.fill_alpha, 
            overlay, 
            self.fill_alpha, 
            0
        )
        
        # Draw outline around silhouette
        if self.outline_px > 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.fill_color, self.outline_px)
        
        return frame
    
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get privacy protection statistics.
        
        Returns:
            Dictionary with privacy configuration stats
        """
        return {
            'enabled': self.enabled,
            'scope': self.scope,
            'face_blur': {
                'enabled': self.face_blur_enabled,
                'method': self.face_blur_method if self.face_blur_enabled else None,
                'detector': 'haar' if self.face_detector is not None else 'bbox_estimation'
            },
            'silhouette': {
                'enabled': self.silhouette_enabled,
                'alpha': self.fill_alpha if self.silhouette_enabled else None
            }
        }


# Standalone function for backward compatibility
def apply_privacy(frame: np.ndarray, 
                 person_data: List[Dict[str, Any]], 
                 weapon_stats: Dict[str, Any],
                 privacy_config: Dict[str, Any]) -> np.ndarray:
    """
    Standalone function to apply privacy protection.
    
    Args:
        frame: Input frame
        person_data: List of person detections
        weapon_stats: Weapon detection results
        privacy_config: Privacy configuration dict
    
    Returns:
        Frame with privacy protection applied
    """
    protector = PrivacyProtector(privacy_config)
    return protector.apply_privacy(frame, person_data, weapon_stats)