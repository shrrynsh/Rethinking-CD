import pandas as pd
import json
import os

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "/teamspace/studios/this_studio/cd_rethink/outputs/mme/mme_eval_results.csv"
GT_FILE = "/teamspace/studios/this_studio/cd_rethink/data/mme/mme_reference.jsonl" # You will handle the path
OUTPUT_CSV = "mme_no_to_yes_results.csv"

# Keys based strictly on your provided JSON schemas
KEY_QID = "question_id"
KEY_MODEL_ANS = "text"   
KEY_GT_ANS = "label"      
KEY_SUBTASK = "category" 

def load_jsonl_to_df(filepath):
    """Loads a JSONL file into a Pandas DataFrame."""
    if not os.path.exists(filepath):
        return None
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def clean_ans(text):
    """Standardizes the text formatting (since no invalid formats exist)."""
    return str(text).lower().strip()

def calculate_r_score_no_to_yes(baseline_df, cd_df, gt_df, subtask_name):
    """Calculates the Semantic Flip Ratio (R) for No -> Yes flips."""
    if baseline_df is None or cd_df is None:
        return "N/A (Missing File)"

    # Filter GT explicitly by the subtask to ensure tight bounding
    subtask_gt = gt_df[gt_df[KEY_SUBTASK].astype(str).str.lower() == str(subtask_name).lower()]
    
    if len(subtask_gt) == 0:
        return "N/A (GT Subtask mismatch)"

    # 1. Merge Baseline and CD predictions on Question ID
    merged_models = pd.merge(
        baseline_df[[KEY_QID, KEY_MODEL_ANS]], 
        cd_df[[KEY_QID, KEY_MODEL_ANS]], 
        on=KEY_QID, 
        suffixes=('_base', '_cd')
    )
    
    # 2. Merge with the filtered Ground Truth
    merged = pd.merge(
        merged_models,
        subtask_gt[[KEY_QID, KEY_GT_ANS]],
        on=KEY_QID
    )
    
    if len(merged) == 0:
        return "N/A (Merge failed)"
    
    # Standardize binary answers
    merged['ans_base'] = merged[f'{KEY_MODEL_ANS}_base'].apply(clean_ans)
    merged['ans_cd'] = merged[f'{KEY_MODEL_ANS}_cd'].apply(clean_ans)
    merged['ans_gt'] = merged[KEY_GT_ANS].apply(clean_ans)

    # 3. Isolate Baseline "No" answers
    baseline_nos = merged[merged['ans_base'] == "no"]
    if len(baseline_nos) == 0:
        return "N/A (No Base 'No's)"

    # 4. Calculate Numerator: Base = No, CD = Yes, GT = Yes (The "Good" Flip)
    good_flips = baseline_nos[(baseline_nos['ans_cd'] == "yes") & (baseline_nos['ans_gt'] == "yes")]
    numerator = len(good_flips)

    # 5. Calculate Denominator: Base = No, CD = Yes, GT = No (The "Bad" Flip)
    bad_flips = baseline_nos[(baseline_nos['ans_cd'] == "yes") & (baseline_nos['ans_gt'] == "no")]
    denominator = len(bad_flips)

    # 6. Calculate Final R Score
    if denominator == 0:
        if numerator > 0:
            return "Infinity" 
        return "N/A (No Flips)"

    r_score = numerator / denominator
    return round(r_score, 4)

def run_experiment():
    """Generates the formatted blocks and exports everything to a CSV."""
    print("Loading datasets...")
    try:
        df_csv = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: Cannot find {CSV_FILE}.")
        return

    gt_df = load_jsonl_to_df(GT_FILE)
    if gt_df is None:
        print(f"Error: Could not load ground truth file.")
        return

    # Exclude aggregate 'total' rows from the CSV
    subtasks = [s for s in df_csv['subtask'].dropna().unique() if 'total' not in s]
    
    baseline_greedy = df_csv[df_csv['method'] == 'baseline (greedy)']
    baseline_sample = df_csv[df_csv['method'] == 'baseline (sample)']

    master_results = [] 

    for subtask in subtasks:
        print(f"\n{'='*60}")
        print(f"SUBTASK: {subtask.upper()} (NO -> YES SHIFT)")
        print(f"{'='*60}")
        
        subtask_data = df_csv[df_csv['subtask'] == subtask]
        subtask_results = []
        
        for _, row in subtask_data.iterrows():
            method_name = row['method']
            accuracy = row['accuracy']
            yes_pct = row['yes_pct']
            answer_file = row['answer_file']
            
            # Match the CD method to its correct baseline type
            is_sample = 'sample' in method_name.lower()
            baseline_row = baseline_sample[baseline_sample['subtask'] == subtask] if is_sample else baseline_greedy[baseline_greedy['subtask'] == subtask]
            
            r_score = "N/A (Baseline)" 
            
            if "baseline" not in method_name.lower() and not baseline_row.empty:
                base_file = baseline_row.iloc[0]['answer_file']
                
                df_base = load_jsonl_to_df(base_file)
                df_cd = load_jsonl_to_df(answer_file)
                
                r_score = calculate_r_score_no_to_yes(df_base, df_cd, gt_df, subtask)

            row_data = {
                "Subtask": subtask,
                "Method": method_name,
                "Accuracy": f"{accuracy:.4f}" if pd.notna(accuracy) else "N/A",
                "R_No_To_Yes": str(r_score),
                "Yes %": f"{yes_pct:.2f}%" if pd.notna(yes_pct) else "N/A"
            }
            
            subtask_results.append(row_data)
            master_results.append(row_data)
            
        display_df = pd.DataFrame(subtask_results).drop(columns=["Subtask"])
        print(display_df.to_string(index=False))

    final_df = pd.DataFrame(master_results)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'='*60}")
    print(f"Experiment Complete. Results exported to: {OUTPUT_CSV}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_experiment()