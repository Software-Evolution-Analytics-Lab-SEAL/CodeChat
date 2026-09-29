import pandas as pd

def compute_multi_language_density(df, role="assistant", output_csv_path=None):
    """
    Compute Multi-language Density (MLD): average number of distinct programming languages
    used per conversation in assistant responses.

    Args:
        df (pd.DataFrame): Combined DataFrame of code snippets with `conversation_id`, `tag_lan`, and `role`.
        role (str): Role to filter on (default: "assistant").
        output_csv_path (str): Optional path to save per-conversation MLD values.
    
    Returns:
        float: The computed MLD value.
    """
    # Step 1: Filter by role
    if role != "all":
        df = df[df["role"] == role]

    # Step 2: Filter out null and 'None' tag_lan
    df = df[df["tag_lan"].notnull() & (df["tag_lan"] != "None")]

    # Step 3: Group by conversation_id and count distinct languages
    mld_series = df.groupby("conversation_id")["tag_lan"].nunique()

    # Step 4: Compute overall Multi-language Density (MLD)
    mld_value = mld_series.mean()
    print(f"Multi-language Density (MLD): {mld_value:.3f}")

    # Optional: Save per-conversation MLD values
    if output_csv_path:
        mld_series.rename("distinct_language_count").to_csv(output_csv_path, index=True)
        print(f"Per-conversation MLD values saved to: {output_csv_path}")

    return mld_value

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Load full assistant/user data (already combined)
    csv_path = "./rq1_results/2_1allcode_lines.csv"
    all_df = pd.read_csv(csv_path)

    # Compute and save MLD
    mld = compute_multi_language_density(
        all_df,
        role="assistant",
        output_csv_path="./rq1_results/2_1_4_mld.csv"
    )
