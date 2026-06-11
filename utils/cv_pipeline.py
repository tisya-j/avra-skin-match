import cv2
import mediapipe as mp
import numpy as np


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

        # Cheek landmarks (left + right)
        self.target_landmarks = [
            205,  # Left cheek
            425   # Right cheek
        ]

    def extract_skin_patches(self, image_path, patch_size=48):
        """
        Extract representative skin patches from cheek regions
        using larger circular sampling zones.

        Args:
            image_path (str): Path to the input image.
            patch_size (int): Size of the square patch to sample.

        Returns:
            all_skin_pixels (np.ndarray): Extracted skin pixels.
            cropped_debug (np.ndarray): Cropped debug image centered around face.
        """
        image = cv2.imread(image_path)

        if image is None:
            return None, None

        h, w, _ = image.shape

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            print(f"⚠️ No face detected in: {image_path}. Skipping.")
            return None, image.copy()

        face_landmarks = results.multi_face_landmarks[0]
        patches = []

        # Debug visualization
        visual_debug_img = image.copy()

        for lm_id in self.target_landmarks:
            landmark = face_landmarks.landmark[lm_id]
            cx = int(landmark.x * w)
            cy = int(landmark.y * h)

            x1 = max(0, cx - patch_size // 2)
            y1 = max(0, cy - patch_size // 2)
            x2 = min(w, cx + patch_size // 2)
            y2 = min(h, cy + patch_size // 2)

            patch = image[y1:y2, x1:x2]

            if patch.size > 0:
                # Circular mask
                mask = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.uint8)
                center = (patch.shape[1] // 2, patch.shape[0] // 2)
                radius = min(patch.shape[0], patch.shape[1]) // 2

                cv2.circle(mask, center, radius, 255, -1)
                circular_pixels = patch[mask == 255]

                if len(circular_pixels) > 0:
                    patches.append(circular_pixels.reshape(-1, 3))

            # Debug drawings (without landmark ID text)
            cv2.rectangle(visual_debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(visual_debug_img, (cx, cy), 4, (0, 0, 255), -1)
            cv2.circle(visual_debug_img, (cx, cy), patch_size // 2, (255, 255, 0), 2)

        if not patches:
            return None, visual_debug_img

        # Crop around face
        xs = [int(lm.x * w) for lm in face_landmarks.landmark]
        ys = [int(lm.y * h) for lm in face_landmarks.landmark]

        padding = 120
        xmin = max(0, min(xs) - padding)
        xmax = min(w, max(xs) + padding)
        ymin = max(0, min(ys) - padding)
        ymax = min(h, max(ys) + padding)

        cropped_debug = visual_debug_img[ymin:ymax, xmin:xmax]
        all_skin_pixels = np.vstack(patches)

        return all_skin_pixels, cropped_debug
