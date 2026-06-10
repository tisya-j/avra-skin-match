import pandas as pd
import numpy as np

class FoundationRecommender:
    def __init__(self, master_db_path="data/Foundation_shades_with_LAB.csv"):
        self.df = pd.read_csv(master_db_path)

        # clean column names
        self.df.columns = self.df.columns.str.strip()

        # ensure numeric LAB
        self.df["L"] = self.df["L"].astype(float)
        self.df["A"] = self.df["A"].astype(float)
        self.df["B"] = self.df["B"].astype(float)

    # -------------------------
    # LAB DISTANCE
    # -------------------------
    def lab_distance(self, L1, A1, B1, L2, A2, B2):
        return np.sqrt((L1 - L2)**2 + (A1 - A2)**2 + (B1 - B2)**2)

    # -------------------------
    # CONFIDENCE FUNCTION
    # -------------------------
    def compute_confidence(self, distance):
        return 1 / (1 + distance)

    # -------------------------
    # OLIVE CORRECTION LAYER
    # -------------------------
    def olive_correction(self, df, undertone):
        if "Olive" not in undertone:
            return df

        # penalize overly warm/orange shades
        df["distance"] += np.where(df["A"] > 3, 2.5, 0)

        # reward neutral/muted shades
        df["distance"] -= np.where(
            df["A"].between(-2, 3),
            1.2,
            0
        )

        # slight olive preference boost
        df["distance"] -= np.where(
            df["undertone"].str.contains("Olive", na=False),
            1.0,
            0
        )

        return df

    # -------------------------
    # MAIN RECOMMENDER
    # -------------------------
    def get_recommendations(self, L, A, B, undertone=None, top_k=3):

        df = self.df.copy()

        # compute LAB distance for every shade
        df["distance"] = df.apply(
            lambda row: self.lab_distance(
                L, A, B,
                row["L"], row["A"], row["B"]
            ),
            axis=1
        )

        # apply olive correction
        if undertone:
            df = self.olive_correction(df, undertone)

        # rank per brand (top-k)
        top = (
            df.sort_values("distance")
              .groupby("brand")
              .head(top_k)
        )

        # ambiguity detection (global, across all brands)
        df_sorted = df.sort_values("distance")
        if len(df_sorted) > 1:
            best = df_sorted.iloc[0]["distance"]
            second = df_sorted.iloc[1]["distance"]
            ambiguity = (second - best) < 1.5
            confidence_gap = round(second - best, 3)
        else:
            ambiguity = False
            confidence_gap = None

        # format output
        matches = {}
        for brand in top["brand"].unique():
            brand_rows = top[top["brand"] == brand]
            matches[brand] = [
                {
                    "shade_id": r["shade_id"],
                    "shade_code": r["shade_code"],
                    "shade_name": r["shade_name"],
                    "distance": round(float(r["distance"]), 2),
                    "confidence": round(self.compute_confidence(float(r["distance"])), 3)
                }
                for _, r in brand_rows.iterrows()
            ]

        return {
            "status": "success",
            "matches": matches,
            "ambiguity": ambiguity,
            "confidence_gap": confidence_gap
        }
