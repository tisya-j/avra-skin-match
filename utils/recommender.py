import pandas as pd
import os


class FoundationRecommender:
    def __init__(
        self,
        master_db_path="data/Foundation_shades.csv",
        lookup_path="data/Recommendation_lookup.csv"
    ):

        self.master_db_path = master_db_path
        self.lookup_path = lookup_path

        # Verify database files exist
        if (
            not os.path.exists(self.master_db_path)
            or not os.path.exists(self.lookup_path)
        ):
            raise FileNotFoundError(
                f"Missing database files. Please verify that "
                f"'{os.path.basename(self.master_db_path)}' and "
                f"'{os.path.basename(self.lookup_path)}' exist "
                f"inside the data/ folder."
            )

        # Load CSVs
        self.master_df = pd.read_csv(self.master_db_path)
        self.lookup_df = pd.read_csv(self.lookup_path)

        # Clean column names
        self.master_df.columns = self.master_df.columns.str.strip()
        self.lookup_df.columns = self.lookup_df.columns.str.strip()

        # Set index for fast lookups
        self.master_df.set_index("shade_id", inplace=True)

    def get_shade_details(self, shade_id):
        """
        Returns full shade information from the master database.
        """

        if pd.isna(shade_id) or shade_id not in self.master_df.index:
            return {
                "brand": "N/A",
                "shade_code": "N/A",
                "shade_name": "No clean match found",
                "notes": ""
            }

        matched_data = self.master_df.loc[shade_id]

        # Handle duplicate shade IDs safely
        if isinstance(matched_data, pd.DataFrame):
            row = matched_data.iloc[0]
        else:
            row = matched_data

        return {
            "brand": row["brand"],
            "shade_code": row["shade_code"],
            "shade_name": row["shade_name"],
            "notes": str(row["notes"]) if pd.notna(row["notes"]) else ""
        }

    def get_recommendations(self, depth, undertone):
        """
        Returns foundation recommendations based on
        predicted depth and undertone.

        Uses exact matching first.
        Falls back to nearest available depth if needed.
        """

        # Exact profile lookup
        matched_row = self.lookup_df[
            (self.lookup_df["depth"].str.strip() == depth)
            &
            (self.lookup_df["undertone"].str.strip() == undertone)
        ]

        # If exact match doesn't exist, use nearest depth
        if matched_row.empty:

            depth_order = [
                "Fair",
                "Light-Medium",
                "Medium",
                "Tan",
                "Deep"
            ]

            if depth not in depth_order:
                return {
                    "status": "error",
                    "message": f"Unknown depth category: {depth}"
                }

            target_idx = depth_order.index(depth)

            same_undertone_rows = self.lookup_df[
                self.lookup_df["undertone"].str.strip() == undertone
            ]

            if same_undertone_rows.empty:
                return {
                    "status": "error",
                    "message": (
                        f"No recommendations available "
                        f"for undertone '{undertone}'"
                    )
                }

            best_row = None
            best_distance = float("inf")

            for _, row in same_undertone_rows.iterrows():

                candidate_depth = str(row["depth"]).strip()

                if candidate_depth not in depth_order:
                    continue

                candidate_idx = depth_order.index(candidate_depth)

                distance = abs(candidate_idx - target_idx)

                if distance < best_distance:
                    best_distance = distance
                    best_row = row

            if best_row is None:
                return {
                    "status": "error",
                    "message": (
                        f"No suitable fallback found for "
                        f"{depth} + {undertone}"
                    )
                }

            row = best_row

            note = (
                f"Exact profile not found. "
                f"Using nearest available depth match "
                f"({row['depth']})."
            )

        else:
            row = matched_row.iloc[0]
            note = ""

        return {
            "status": "success",
            "profile": {
                "depth": depth,
                "undertone": undertone
            },
            "matches": {
                "Kay Beauty": self.get_shade_details(
                    row["kay_beauty_match"]
                ),
                "Maybelline Fit Me": self.get_shade_details(
                    row["maybelline_match"]
                ),
                "Lakme Powerplay": self.get_shade_details(
                    row["lakme_match"]
                )
            },
            "note": note
        }


if __name__ == "__main__":

    try:
        recommender = FoundationRecommender()

        test_depth = "Medium"
        test_undertone = "Olive"

        print(
            f"Testing lookup logic for: "
            f"{test_depth} skin with "
            f"{test_undertone} undertones..."
        )

        results = recommender.get_recommendations(
            test_depth,
            test_undertone
        )

        if results["status"] == "success":

            print("\nRecommendations found:\n")

            for brand, specs in results["matches"].items():

                print(
                    f"🔹 {brand}: "
                    f"{specs['shade_code']} "
                    f"({specs['shade_name']})"
                )

            if results["note"]:
                print("\nFallback Note:")
                print(results["note"])

        else:
            print(results["message"])

    except Exception as e:
        print(f"Error executing recommender: {e}")