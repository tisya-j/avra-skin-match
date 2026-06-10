import gradio as gr
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

# Import our custom backend modules
from utils.cv_pipeline import SkinExtractor
from utils.colour_analyzer import ColourAnalyzer
from utils.recommender import FoundationRecommender

# Initialize our core classes
extractor = SkinExtractor()
analyzer = ColourAnalyzer()
recommender = FoundationRecommender()

def estimate_true_skin_color(pixels):
    """
    Removes noise + finds dominant skin cluster
    """
    if pixels is None or len(pixels) == 0:
        return None

    pixels = pixels.reshape(-1, 3)

    # Step 1: brightness filter
    brightness = np.mean(pixels, axis=1)
    low = np.percentile(brightness, 20)
    high = np.percentile(brightness, 80)
    filtered = pixels[(brightness >= low) & (brightness <= high)]

    if len(filtered) < 10:
        filtered = pixels  # fallback

    # Step 2: KMeans clustering
    k = min(3, len(filtered))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(filtered)

    # pick biggest cluster (most likely skin)
    counts = np.bincount(labels)
    dominant_cluster = np.argmax(counts)
    skin_color = kmeans.cluster_centers_[dominant_cluster]

    return skin_color  # BGR

def bgr_to_hex_safe(bgr):
    b, g, r = bgr
    b = int(np.clip(b, 0, 255))
    g = int(np.clip(g, 0, 255))
    r = int(np.clip(r, 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"

def generate_insights(undertone):
    """Generates Card 3 educational copy based on the classified undertone"""
    if "Olive" in undertone:
        return (
            "About Olive Undertones: Your skin has a distinct greenish/neutral hue, which is common "
            "in India but frequently ignored by global brands. Standard warm foundations will often look way too orange "
            "on you, while cool shades can make you look gray. Look for shades with neutral, muted, or explicit olive descriptions."
        )
    elif undertone == "Neutral Warm":
        return (
            "About Neutral Warm Undertones: Your skin sits in a delicate, muted transition zone—it carries a soft hint "
            "of warmth without pulling heavily golden or yellow. Standard warm/honey shades will often oxidize or look "
            "too orange on you. Sticking to neutral bases with just a touch of brightness keeps your complexion looking clear and natural."
        )
    elif "Warm" in undertone:
        return (
            "About Warm Undertones: Your skin has golden or yellow baseline tones. Most commercial "
            "Indian ranges cater well to this profile. Golden and honey-toned pigments will blend seamlessly into your skin."
        )
    else:
        return (
            "About Neutral/Cool Undertones: Your skin carries a balanced mix of pink and golden tones, or leans "
            "slightly rosy. Avoid overly yellow or orange base products to keep your natural complexion bright and clear."
        )

def process_pipeline(input_files):
    if input_files is None or len(input_files) == 0:
        return "Warning: Please upload at least one selfie for calibration!", "", "", "", None

    lab_list = []
    valid_debug_imgs = []
    pooled_pixels = []

    for file_obj in input_files:
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        img = cv2.imread(file_path)
        if img is None:
            continue

        pixels, debug_img = extractor.extract_skin_patches(file_path)
        if pixels is None:
            continue

        pooled_pixels.append(pixels)
        lab_profile = analyzer.process_skin(pixels)
        lab_list.append([lab_profile["L"], lab_profile["A"], lab_profile["B"]])

        if debug_img is not None:
            rgb_debug = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            valid_debug_imgs.append(rgb_debug)

    if not lab_list:
        return (
            "Error: Face Not Detected\n\nPlease ensure your face is fully visible, well-lit, and facing the camera directly.",
            "", "", "", None
        )

    try:
        lab_array = np.array(lab_list)
        final_L = np.median(lab_array[:, 0])
        final_A = np.median(lab_array[:, 1])
        final_B = np.median(lab_array[:, 2])

        # Build profile with aggregated LAB
        profile = {
            "L": final_L,
            "A": final_A,
            "B": final_B,
            "undertone": lab_profile["undertone"],
            "depth": lab_profile["depth"]
        }

        print(f"L={final_L:.1f}, A={final_A:.1f}, B={final_B:.1f}")

        # Updated recommender call
        rec_data = recommender.get_recommendations(
            profile["L"],
            profile["A"],
            profile["B"],
            profile["undertone"]
        )

        # ✅ New swatch logic
        master_pixels = np.vstack(pooled_pixels)
        skin_bgr = estimate_true_skin_color(master_pixels)
        if skin_bgr is not None:
            hex_color = bgr_to_hex_safe(skin_bgr)
        else:
            hex_color = "#cccccc"

        swatch_html = f"""
        <div style='display: flex; align-items: center; gap: 12px; margin-top: 10px;'>
            <div style='width: 45px; height: 45px; background-color: {hex_color}; border-radius: 8px; border: 2px solid #ddd;'></div>
            <p style='margin: 0; font-size: 14px;'><b>Estimated Skin Swatch:</b> {hex_color}</p>
        </div>
        """

        card1_md = f"""
        ### Card 1 -- Skin Profile
        * Skin Depth: {profile['depth']} (L: {profile['L']:.1f})
        * Undertone: {profile['undertone']} (A: {profile['A']:.1f} | B: {profile['B']:.1f})
        """

        if rec_data["status"] == "success":
            matches = rec_data["matches"]
            card2_md = "### Card 2 -- Foundation Matches\n\n"
            for brand, shades in matches.items():
                card2_md += f"**{brand}**\n"
                for s in shades:
                    card2_md += (
                        f"- {s['shade_code']} -- {s['shade_name']} "
                        f"(dist: {s['distance']}, conf: {s['confidence']})\n"
                    )
                card2_md += "\n"

            if rec_data.get("ambiguity"):
                card2_md += "\n\n⚠️ **Ambiguous match detected:** You are between two foundation tones."
        else:
            card2_md = f"### Card 2 -- Foundation Matches\n\nWarning: {rec_data['message']}"

        card3_md = f"### Card 3 -- Undertone Explanation\n\n{generate_insights(profile['undertone'])}"
        display_preview = valid_debug_imgs[0] if valid_debug_imgs else None

        return card1_md, swatch_html, card2_md, card3_md, display_preview

    except Exception as e:
        return f"Error: Backend Pipeline Failure: {str(e)}", "", "", "", None

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# Avra")
    gr.Markdown("### An AI-powered computer vision foundation shade matcher engineered specifically for Indian skin tones.")
    
    with gr.Row():
        with gr.Column(scale=1):
            user_images = gr.File(file_count="multiple", file_types=["image"], label="Upload 2-4 Selfies together")
            submit_btn = gr.Button("Calibrate My True Skin Tone", variant="primary")
            gr.Markdown("### Face Detection Tracking Mesh")
            debug_preview = gr.Image(label="MediaPipe Sample Regions (First Photo)", interactive=False)

        with gr.Column(scale=1):
            output_card1 = gr.Markdown("### Card 1 -- Skin Profile\nAwaiting your photo inputs...")
            swatch_preview = gr.HTML("")
            output_card2 = gr.Markdown("### Card 2 -- Foundation Matches\nAwaiting structural extraction results...")
            output_card3 = gr.Markdown("### Card 3 -- Undertone Explanation\nAwaiting skin analysis parameters...")

    submit_btn.click(
        fn=process_pipeline,
        inputs=[user_images],
        outputs=[output_card1, swatch_preview, output_card2, output_card3, debug_preview]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
