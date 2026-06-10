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

        # Central forehead + under-eye cheek zones
        self.target_landmarks = [
              # Forehead
            205,  # Left cheek
            425   # Right cheek
        ]

    def extract_skin_patches(self, image_path, patch_size=48):
        """
        Extract representative skin patches from
        forehead and central cheek regions using
        larger circular sampling zones.
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

                mask = np.zeros(
                    (patch.shape[0], patch.shape[1]),
                    dtype=np.uint8
                )

                center = (
                    patch.shape[1] // 2,
                    patch.shape[0] // 2
                )

                radius = min(
                    patch.shape[0],
                    patch.shape[1]
                ) // 2

                cv2.circle(
                    mask,
                    center,
                    radius,
                    255,
                    -1
                )

                circular_pixels = patch[mask == 255]

                if len(circular_pixels) > 0:
                    patches.append(
                        circular_pixels.reshape(-1, 3)
                    )

            # Draw sampling box
            cv2.rectangle(
                visual_debug_img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw center point
            cv2.circle(
                visual_debug_img,
                (cx, cy),
                4,
                (0, 0, 255),
                -1
            )

            # Draw actual sampling circle
            cv2.circle(
                visual_debug_img,
                (cx, cy),
                patch_size // 2,
                (255, 255, 0),
                2
            )

            # Show landmark ID
            cv2.putText(
                visual_debug_img,
                str(lm_id),
                (cx + 5, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        if not patches:
            return None, visual_debug_img

        all_skin_pixels = np.vstack(patches)

        return all_skin_pixels, visual_debug_img