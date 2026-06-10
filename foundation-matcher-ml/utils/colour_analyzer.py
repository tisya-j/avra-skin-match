import cv2
import numpy as np

class ColourAnalyzer:
    def __init__(self):
        # Optimized LAB Reference Points 
        # Fine-tuned for true light/light-medium Indian skin tone variations (Muted/Olive/Neutral)
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
        Takes raw BGR skin pixels, calculates a stable median, scales it,
        and applies an illumination-guard logic to reverse indoor camera shadows.
        """
        if skin_pixels is None or len(skin_pixels) == 0:
            raise ValueError("Skin pixel matrix is empty or invalid.")

        # Compute stable median across the entire multi-photo pool cleanly
        median_bgr = np.round(np.median(skin_pixels, axis=0)).astype(np.uint8)
        
        # Convert to LAB space via OpenCV
        bgr_pixel_img = np.uint8([[median_bgr]])
        lab_pixel_img = cv2.cvtColor(bgr_pixel_img, cv2.COLOR_BGR2LAB)
        l_val, a_val, b_val = lab_pixel_img[0][0]
        
        # Decode standard 0-100 L and -128 to 127 A/B values
        standard_l = (l_val / 255.0) * 100
        standard_a = float(a_val) - 128
        standard_b = float(b_val) - 128
        
        # 🛡️ THE LIGHTING GUARD ENGINE:
        # If L is crushed below 62 due to indoor room lighting or vehicle shadows,
        # apply a progressive calibration scale factor to restore true baseline skin depth.
        if standard_l < 62.0:
            shadow_deficit = 62.0 - standard_l
            # Progressively lift lightness based on severity of the shadow lock
            standard_l += (shadow_deficit * 0.85) + 6.0
            
            # Dampen background clothing flush adjustments on chromatic channels
            standard_a *= 0.55
            standard_b *= 0.65

        return float(standard_l), float(standard_a), float(standard_b)

    def classify_depth(self, l_val):
        """
        Maps corrected Lightness values directly to accurate skin depths.
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
        closest_undertone = None
        min_distance = float('inf')
        
        for name, coords in self.undertone_references.items():
            distance = ((a_val - coords["A"])**2 + (b_val - coords["B"])**2)**0.5
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
    mock_pixels = np.array([[115, 145, 160]] * 400) 
    results = analyzer.process_skin(mock_pixels)
    print(" Testing colour_analyzer.py pipeline:")
    print(f"Calculated LAB Values -> L: {results['L']}, A: {results['A']}, B: {results['B']}")
    print(f"Predicted Profile    -> Depth: {results['depth']} | Undertone: {results['undertone']}")