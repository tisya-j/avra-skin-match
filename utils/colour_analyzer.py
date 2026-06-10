import cv2
import numpy as np

class ColourAnalyzer:
    def __init__(self):
        # LAB reference points for undertone classification
        self.undertone_references = {
            "Cool Pink":      {"A": 6.0,  "B": 4.0},
            "Cool Neutral":   {"A": 3.5,  "B": 7.0},
            "Neutral Muted":  {"A": 1.5,  "B": 10.0},
            "Neutral Warm":   {"A": 2.5,  "B": 13.0},
            "Warm Golden":    {"A": 4.0,  "B": 18.0},
            "Olive":          {"A": -1.5, "B": 9.0},
            "Warm Olive":     {"A": -0.5, "B": 14.0}
        }

    def get_dominant_lab(self, skin_pixels):
        """
        Computes the dominant skin color from pooled skin pixels
        and converts it to standard LAB coordinates.
        """

        if skin_pixels is None or len(skin_pixels) == 0:
            raise ValueError("Skin pixel matrix is empty or invalid.")

        # Stable median color across all sampled skin regions
        median_bgr = np.round(
            np.median(skin_pixels, axis=0)
        ).astype(np.uint8)

        # Convert median pixel to LAB
        bgr_pixel_img = np.uint8([[median_bgr]])
        lab_pixel_img = cv2.cvtColor(
            bgr_pixel_img,
            cv2.COLOR_BGR2LAB
        )

        l_val, a_val, b_val = lab_pixel_img[0][0]

        # Convert OpenCV LAB ranges
        standard_l = (float(l_val) / 255.0) * 100.0
        standard_a = float(a_val) - 128.0
        standard_b = float(b_val) - 128.0

        # Conservative shadow correction
        # Only applies to very dark captures where lighting
        # is likely crushing the true skin value.
        if standard_l < 45:
            standard_l += 2

        # Debugging output (optional)
        print(
            f"Median BGR={median_bgr} | "
            f"L={standard_l:.1f}, "
            f"A={standard_a:.1f}, "
            f"B={standard_b:.1f}"
        )

        return standard_l, standard_a, standard_b

    def classify_depth(self, l_val):
        """
        Maps LAB lightness values to skin depth categories.
        """

        if l_val >= 70:
            return "Fair"
        elif l_val >= 61:
            return "Light-Medium"
        elif l_val >= 48:
            return "Medium"
        elif l_val >= 36:
            return "Tan"
        else:
            return "Deep"

    def classify_undertone(self, a_val, b_val):
        """
        Finds nearest undertone anchor using Euclidean distance.
        """

        closest_undertone = None
        min_distance = float("inf")

        for name, coords in self.undertone_references.items():

            distance = (
                (a_val - coords["A"]) ** 2 +
                (b_val - coords["B"]) ** 2
            ) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_undertone = name

        return closest_undertone

    def process_skin(self, skin_pixels):

        L, A, B = self.get_dominant_lab(skin_pixels)

        depth = self.classify_depth(L)
        undertone = self.classify_undertone(A, B)

        return {
            "L": round(L, 1),
            "A": round(A, 1),
            "B": round(B, 1),
            "depth": depth,
            "undertone": undertone
        }


if __name__ == "__main__":

    analyzer = ColourAnalyzer()

    mock_pixels = np.array(
        [[115, 145, 160]] * 400
    )

    results = analyzer.process_skin(mock_pixels)

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
        f"Undertone: {results['undertone']}"
    )