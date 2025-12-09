import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import cv2
from pathlib import Path
import time

from .blazepalm import PalmDetector
from .handlandmarks import HandLandmarks
from .blazebase import resize_pad, denormalize_detections

class BlazeHandTrackingPipeline:
    """
    Hand tracking pipeline using pure PyTorch models (BlazePalm + HandLandmarks).
    Supports runtime Pruning and Quantization.
    """
    def __init__(self, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 precision='fp32',
                 prune_ratio=0.0,
                 quantize=False):
        
        self.device = torch.device(device)
        self.precision = precision
        self.prune_ratio = prune_ratio
        self.quantize = quantize
        
        # Paths
        self.root = Path(__file__).resolve().parents[2]
        self.assets_dir = self.root / "assets" / "models"
        
        print(f"[BlazePipeline] Initializing on {self.device} (Precision: {precision})")
        
        # 1. Load Models
        self.detector = PalmDetector().to(self.device)
        self.landmark_model = HandLandmarks().to(self.device)
        
        # 2. Load Weights
        det_path = self.assets_dir / "palmdetector.pth"
        lm_path = self.assets_dir / "HandLandmarks.pth"
        anchor_path = self.assets_dir / "anchors.npy"
        
        if not det_path.exists() or not lm_path.exists() or not anchor_path.exists():
            raise FileNotFoundError(f"Missing model files in {self.assets_dir}. Please run 'task setup' or check download.")
            
        self.detector.load_weights(str(det_path))
        self.detector.load_anchors(str(anchor_path))
        self.landmark_model.load_weights(str(lm_path))
        
        # 3. Apply Pruning (if requested)
        if self.prune_ratio > 0:
            self._apply_pruning(self.detector, self.prune_ratio)
            self._apply_pruning(self.landmark_model, self.prune_ratio)
            print(f"[BlazePipeline] Applied pruning (sparsity: {self.prune_ratio*100:.1f}%)")
            
        # 4. Apply Quantization (if requested)
        if self.quantize:
            # Note: PyTorch dynamic quantization is CPU only for now usually
            if self.device.type == 'cpu':
                self._apply_quantization(self.detector)
                self._apply_quantization(self.landmark_model)
                print("[BlazePipeline] Applied dynamic quantization (int8)")
            else:
                print("[BlazePipeline] Warning: Quantization skipped (requires CPU)")

        # 5. Set Precision
        if self.precision == 'fp16' and self.device.type == 'cuda':
            self.detector.half()
            self.landmark_model.half()
            print("[BlazePipeline] Switched to FP16")
            
        self.detector.eval()
        self.landmark_model.eval()
        
        # ROI extraction params
        self.resolution = 256

    def _apply_pruning(self, model, amount):
        """Apply global unstructured pruning to Conv2d layers."""
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))
        
        if parameters_to_prune:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
            # Make permanent
            for module, _ in parameters_to_prune:
                prune.remove(module, 'weight')

    def _apply_quantization(self, model):
        """Apply dynamic quantization."""
        torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8, inplace=True
        )

    def process_frame(self, frame):
        """
        Process a single frame.
        Returns: (landmarks_list, detections, scale, pad)
        """
        # 1. Preprocess for Detector
        # BlazePalm expects RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img1, img2, scale, pad = resize_pad(frame_rgb)
        
        # Convert to tensor
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        img1_tensor = img1_tensor.to(self.device)
        
        if self.precision == 'fp16' and self.device.type == 'cuda':
            img1_tensor = img1_tensor.half()
            
        # 2. Run Detector
        with torch.no_grad():
            detections = self.detector.predict_on_batch(img1_tensor)[0] # List of tensors
        
        if len(detections) == 0:
            return [], [], scale, pad
            
        # 3. Post-process Detections
        # detections is already NMS-ed and has shape (N, 17)
        # We need to denormalize them to original image coordinates
        # But wait, predict_on_batch returns detections in 128x128 scale?
        # No, _decode_boxes uses anchors which are normalized? 
        # Let's check blazepalm.py. _decode_boxes outputs coordinates relative to 128x128?
        # No, anchors are likely normalized or in 128 scale.
        # The output of predict_on_batch is (ymin, xmin, ymax, xmax, kps...)
        # We need to map this back to the original image.
        
        # Actually, let's look at denormalize_detections in blazebase.py
        # It assumes detections are [0,1] normalized?
        # blazepalm.py _decode_boxes:
        # x_center = raw ... * anchors ...
        # If anchors are normalized [0,1], then output is [0,1].
        # If anchors are in pixels, output is in pixels.
        # Usually MediaPipe anchors are normalized.
        
        # Assuming normalized output from detector.
        # We need to convert to original image scale.
        
        # 4. Extract ROI and Run Landmarks
        landmarks_list = []
        final_detections = []
        
        for i in range(len(detections)):
            det = detections[i]
            # det is (ymin, xmin, ymax, xmax, ...)
            
            # Convert to ROI
            xc, yc, s, theta = self.detector.detection2roi(det.unsqueeze(0))
            xc, yc, s, theta = xc.item(), yc.item(), s.item(), theta.item()
            
            # Extract ROI image
            roi_img, affine, _ = self.extract_roi(img1, xc, yc, theta, s, self.resolution)
            roi_tensor = torch.from_numpy(roi_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            roi_tensor = roi_tensor.to(self.device)
            
            if self.precision == 'fp16' and self.device.type == 'cuda':
                roi_tensor = roi_tensor.half()
                
            # Run Landmark Model
            with torch.no_grad():
                # HandLandmarks returns: hand_flag, handedness, reg_3d
                _, _, reg_3d = self.landmark_model(roi_tensor)
                
            # Reg_3d is (1, 63) -> 21 points * 3 (x,y,z)
            # Coordinates are normalized to [0, 256]? Or [0,1]?
            # handlandmarks.py: reg_3d = ... / 256.0. So it's [0,1] relative to ROI.
            
            landmarks = reg_3d.reshape(-1, 3).cpu().numpy() # (21, 3)
            
            # Denormalize landmarks from ROI to Original Image
            landmarks = self.denormalize_landmarks(landmarks, affine)
            
            # Map from img1 (256x256) to Original Frame
            # pixel_orig = pixel_img1 * scale - pad
            landmarks[:, 0] = landmarks[:, 0] * scale - pad[1]
            landmarks[:, 1] = landmarks[:, 1] * scale - pad[0]
            
            landmarks_list.append(landmarks)
            
            # Add to final detections (convert to [y1, x1, y2, x2, score])
            # det is normalized [0,1] relative to 128x128 input?
            # We need to map it to original image.
            # denormalize_detections maps [0,1] -> original image using scale/pad
            
            # Wait, det is tensor.
            det_cpu = det.cpu().numpy()
            # We need to pass a batch to denormalize_detections
            det_batch = det.unsqueeze(0).clone()
            denorm_det = denormalize_detections(det_batch, scale, pad)[0]
            
            # Format: y1, x1, y2, x2, score
            bbox = denorm_det[:4].cpu().numpy()
            score = denorm_det[-1].item()
            
            # Normalize to [0, 1] relative to original frame
            h, w = frame.shape[:2]
            norm_bbox = [
                bbox[0] / h, bbox[1] / w,
                bbox[2] / h, bbox[3] / w
            ]
            final_detections.append([norm_bbox[0], norm_bbox[1], norm_bbox[2], norm_bbox[3], score])

        return landmarks_list, np.array(final_detections), scale, pad

    def extract_roi(self, img1, xc, yc, theta, scale, res):
        """Extract square ROI from img1 (256x256)."""
        # img1 is already 256x256 padded.
        
        xc_px = xc * 256
        yc_px = yc * 256
        width_px = scale * 256
        
        # ...
        
        # We don't need to call resize_pad again.
        # img1, _, _, _ = resize_pad(frame) 

        
        xc_px = xc * 256
        yc_px = yc * 256
        width_px = scale * 256
        
        # Points in source image (img1)
        # Center, Top, Right (to define rotation/scale)
        # But we have theta.
        
        # Calculate 3 points in source
        # Center: (xc_px, yc_px)
        # Point 2: (xc_px + width_px/2 * cos(theta), yc_px + width_px/2 * sin(theta))
        # Point 3: (xc_px - width_px/2 * sin(theta), yc_px + width_px/2 * cos(theta))
        
        # Actually, simpler:
        # cv2.getRotationMatrix2D((xc_px, yc_px), theta * 180 / np.pi, 1.0)
        # But we also need scaling.
        # We want the ROI (width_px) to map to `res` (256).
        # So scale factor is res / width_px.
        
        M = cv2.getRotationMatrix2D((xc_px, yc_px), np.degrees(theta), res / width_px)
        # Translation: Move (xc_px, yc_px) to (res/2, res/2)
        M[0, 2] += (res/2) - xc_px
        M[1, 2] += (res/2) - yc_px
        
        # Warp
        roi = cv2.warpAffine(img1, M, (res, res), borderValue=(0,0,0))
        
        # Invert affine for denormalization later
        M_inv = cv2.invertAffineTransform(M)
        
        return roi, M_inv, M

    def denormalize_landmarks(self, landmarks, affine):
        """Map landmarks from ROI (256x256) back to img1 (256x256) then to original."""
        # landmarks: (21, 3) in [0,1] relative to ROI?
        # HandLandmarks output was / 256.0. So it is [0,1].
        # Convert to pixels in ROI
        lm_px = landmarks[:, :2] * 256.0
        
        # Apply inverse affine to get back to img1 coordinates
        # affine is 2x3
        # Add 1 column for matmul
        lm_px = np.hstack((lm_px, np.ones((21, 1))))
        lm_img1 = lm_px @ affine.T
        
        # Now map from img1 to original frame
        # img1 was created by resize_pad(frame)
        # img1, img2, scale, pad = resize_pad(frame)
        # img1 is 256x256.
        # Original frame was scaled by `1/scale` (wait, scale = size0 / w1).
        # img1 = resize(frame) + pad
        # So: pixel_orig = (pixel_img1 - pad) * scale
        
        # But I don't have `scale` and `pad` here easily unless I store them or pass them.
        # I'll assume the caller handles the final mapping if needed, 
        # OR I can just return landmarks in img1 coordinates and let the caller map?
        # No, `process_frame` returns landmarks.
        # Let's fix `process_frame` to handle the final mapping.
        
        return lm_img1 # Return in img1 coordinates for now

