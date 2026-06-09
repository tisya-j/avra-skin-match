import cv2
import mediapipe as mp
import numpy as np
import mediapipe.solutions.face_mesh as mp_face_mesh

class SkinExtractor:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Exact landmarks for core skin sampling zones
        self.target_landmarks = [117, 346, 215, 435]

    def extract_skin_patches(self, image_path, patch_size=20):
        """
        Loads an image, finds facial landmarks, and crops out skin patches.
        Returns None if no face is detected instead of crashing.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None, None
            
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            print(f"⚠️ No face detected in: {image_path}. Skipping this frame.")
            return None, image.copy()

        face_landmarks = results.multi_face_landmarks[0]
        patches = []
        visual_debug_img = image.copy()

        for lm_id in self.target_landmarks:
            landmark = face_landmarks.landmark[lm_id]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            
            # Boundary safety check
            x1 = max(0, cx - patch_size // 2)
            y1 = max(0, cy - patch_size // 2)
            x2 = min(w, cx + patch_size // 2)
            y2 = min(h, cy + patch_size // 2)
            
            patch = image[y1:y2, x1:x2]
            if patch.size > 0:
                patches.append(patch)
            
            # Draw boundaries and center anchors for full visual visibility
            cv2.rectangle(visual_debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(visual_debug_img, (cx, cy), 2, (0, 0, 255), -1)

        if not patches:
            return None, visual_debug_img

        all_skin_pixels = np.vstack([p.reshape(-1, 3) for p in patches])
        return all_skin_pixels, visual_debug_img