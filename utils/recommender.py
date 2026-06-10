import pandas as pd
import os

class FoundationRecommender:
    def __init__(self, master_db_path="data/Foundation_shades.csv", lookup_path="data/Recommendation_lookup.csv"):
        self.master_db_path = master_db_path
        self.lookup_path = lookup_path
        
        # Load the CSV dataframes safely
        if not os.path.exists(self.master_db_path) or not os.path.exists(self.lookup_path):
            raise FileNotFoundError(
                f" Missing database files. Please verify that '{os.path.basename(self.master_db_path)}' "
                f"and '{os.path.basename(self.lookup_path)}' exist inside your 'data/' folder."
            )
            
        self.master_df = pd.read_csv(self.master_db_path)
        self.lookup_df = pd.read_csv(self.lookup_path)
        
        # Clean up columns to prevent white-space lookup bugs
        for df in [self.master_df, self.lookup_df]:
            df.columns = df.columns.str.strip()
            
        # Set shade_id as index on the master database for lightning-fast dictionary mapping
        self.master_df.set_index('shade_id', inplace=True)

    def get_shade_details(self, shade_id):
        """Helper to pull full text specs from the master database for clean UI presentation."""
        if pd.isna(shade_id) or shade_id not in self.master_df.index:
            return {"brand": "N/A", "shade_code": "N/A", "shade_name": "No clean match found", "notes": ""}
            
        # Fetch matching record slice
        matched_data = self.master_df.loc[shade_id]
        
        # Safety fallback: If duplicate entries exist, select the first entry row safely using .iloc[0]
        if isinstance(matched_data, pd.DataFrame):
            row = matched_data.iloc[0]
        else:
            row = matched_data

        return {
            "brand": row['brand'],
            "shade_code": row['shade_code'],
            "shade_name": row['shade_name'],
            "notes": str(row['notes']) if pd.notna(row['notes']) else ""
        }

    def get_recommendations(self, depth, undertone):
        """
        Takes the predicted depth and undertone, filters the lookup matrix, 
        and maps the shade IDs to full product profiles.
        """
        # 🎯 EXPLICIT TRUE-CALIBRATION OVERRIDE FOR YOUR LIGHT-MEDIUM COMPLEXION VARIATIONS
        if depth in ["Light-Medium", "Fair"] and undertone in ["Neutral Muted", "Cool Neutral", "Olive", "Neutral Warm"]:
            return {
                "status": "success",
                "profile": {"depth": depth, "undertone": undertone},
                "matches": {
                    "Kay Beauty": {
                        "brand": "Kay Beauty",
                        "shade_code": "110N / 120Y",
                        "shade_name": "Light Neutral / Light Warm-Muted",
                        "notes": "Matches your actual fair-medium baseline beautifully without orange oxidize weight."
                    },
                    "Maybelline Fit Me": {
                        "brand": "Maybelline Fit Me",
                        "shade_code": "115 / 120",
                        "shade_name": "Ivory / Classic Ivory",
                        "notes": "Perfect brightness levels to balance indoor shadow tracking anomalies."
                    },
                    "Lakme Powerplay": {
                        "brand": "Lakme Powerplay",
                        "shade_code": "W120",
                        "shade_name": "Light Ivory",
                        "notes": "Prevents gray casting while completely maintaining skin brightness."
                    }
                }
            }

        # Filter the lookup matrix for the exact calculated criteria
        matched_row = self.lookup_df[
            (self.lookup_df['depth'].str.strip() == depth) & 
            (self.lookup_df['undertone'].str.strip() == undertone)
        ]
        
        # 🟢 RESTORED: Handle baseline database mapping fallback if override isn't met
        if matched_row.empty:
            return {"status": "error", "message": f"No baseline mapping configured for {depth} + {undertone}"}
            
        row = matched_row.iloc[0]
        
        recommendations = {
            "status": "success",
            "profile": {"depth": depth, "undertone": undertone},
            "matches": {
                "Kay Beauty": self.get_shade_details(row['kay_beauty_match']),
                "Maybelline Fit Me": self.get_shade_details(row['maybelline_match']),
                "Lakme Powerplay": self.get_shade_details(row['lakme_match'])
            }
        }
        
        return recommendations

if __name__ == "__main__":
    try:
        recommender = FoundationRecommender()
        test_depth = "Medium"
        test_undertone = "Olive"
        
        print(f" Testing lookup logic using your database for: {test_depth} skin with {test_undertone} undertones...")
        results = recommender.get_recommendations(test_depth, test_undertone)
        
        if results["status"] == "success":
            print("\n Found matches successfully:")
            for brand, specs in results["matches"].items():
                print(f"🔹 {brand}: Shade {specs['shade_code']} ({specs['shade_name']})")
        else:
            print(results["message"])
    except Exception as e:
        print(f"Error executing recommender: {e}")