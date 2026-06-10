import cv2
import numpy as np


class ColourAnalyzer:
    def __init__(self):

        # LAB undertone anchors
        self.undertone_references = {

            "Cool Pink": {
                "A": 6.0,
                "B": 4.0
            },

            "Cool Neutral": {
                "A": 3.5,
                "B": 7.0
            },

            "Neutral": {
                "A": 1.5,
                "B": 10.0
            },

            "Neutral Warm": {
                "A": 2.5,
                "B": 13.0
            },

            "Warm Golden": {
                "A": 4.0,
                "B": 18.0
            },

            "Olive": {
                "A": -1.5,
                "B": 9.0
            },

            "Warm Olive": {
                "A": -0.5,
                "B": 14.0
            }
        }

    def get_dominant_lab(self, skin_pixels):
        """
        Computes dominant skin LAB values while rejecting
        extreme shadows and highlights.
        """

        if skin_pixels is None or len(skin_pixels) == 0:
            raise ValueError("Skin pixel matrix is empty or invalid.")

        # ------------------------------------------
        # Remove extreme highlights and shadows
        # ------------------------------------------

        brightness = np.mean(skin_pixels, axis=1)

        low_cut = np.percentile(brightness, 20)
        high_cut = np.percentile(brightness, 80)

        filtered_pixels = skin_pixels[
            (brightness >= low_cut) &
            (brightness <= high_cut)
        ]

        if len(filtered_pixels) == 0:
            filtered_pixels = skin_pixels

        # ------------------------------------------
        # Convert ALL filtered pixels to LAB
        # ------------------------------------------

        lab_pixels = cv2.cvtColor(
            filtered_pixels.reshape(-1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2LAB
        ).reshape(-1, 3)

        l_val = np.median(lab_pixels[:, 0])
        a_val = np.median(lab_pixels[:, 1])
        b_val = np.median(lab_pixels[:, 2])

        # ------------------------------------------
        # Convert OpenCV LAB -> Standard LAB
        # ------------------------------------------

        standard_l = (float(l_val) / 255.0) * 100.0
        standard_a = float(a_val) - 128.0
        standard_b = float(b_val) - 128.0

        # ------------------------------------------
        # Calibration
        # ------------------------------------------

        standard_l -= 4

        if standard_l < 45:
            standard_l += 2

        print(
            f"L={standard_l:.1f}, "
            f"A={standard_a:.1f}, "
            f"B={standard_b:.1f}"
        )

        return standard_l, standard_a, standard_b

    def classify_depth(self, l_val):
        """
        Depth buckets aligned to foundation mapping CSV.
        Softer transitions than the original version.
        """

        if l_val >= 76:
            return "Fair"

        elif l_val >= 68:
            return "Light"

        elif l_val >= 60:
            return "Light-Medium"

        elif l_val >= 52:
            return "Medium"

        elif l_val >= 44:
            return "Medium-Tan"

        elif l_val >= 38:
            return "Tan"

        else:
            return "Deep"

    def classify_undertone(self, a_val, b_val):
        """
        Weighted LAB distance.
        Yellow axis (B) matters more than red-green axis (A).
        """

        closest_undertone = None
        min_distance = float("inf")

        for name, coords in self.undertone_references.items():

            distance = (
                ((a_val - coords["A"]) * 1.0) ** 2 +
                ((b_val - coords["B"]) * 1.5) ** 2
            ) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_undertone = name

        confidence = max(
            0,
            min(
                100,
                100 - (min_distance * 8)
            )
        )

        return closest_undertone, confidence

    def process_skin(self, skin_pixels):

        L, A, B = self.get_dominant_lab(skin_pixels)

        depth = self.classify_depth(L)

        undertone, undertone_confidence = (
            self.classify_undertone(A, B)
        )

        return {
            "L": round(L, 1),
            "A": round(A, 1),
            "B": round(B, 1),

            "depth": depth,

            "undertone": undertone,
            "undertone_confidence": round(
                undertone_confidence,
                1
            )
        }


if __name__ == "__main__":

    analyzer = ColourAnalyzer()

    mock_pixels = np.array(
        [[115, 145, 160]] * 400,
        dtype=np.uint8
    )

    results = analyzer.process_skin(
        mock_pixels
    )

    print("\nTesting colour_analyzer.py pipeline:")

    print(
        f"Calculated LAB Values -> "
        f"L: {results['L']}, "
        f"A: {results['A']}, "
        f"B: {results['B']}"
    )

    print(
        f"Predicted Profile -> "
        f"Depth: {results['depth']} | "
        f"Undertone: {results['undertone']} "
        f"({results['undertone_confidence']}%)"
    )