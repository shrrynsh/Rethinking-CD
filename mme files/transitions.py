import pandas as pd
import json
import os

# ==========================================
# CONFIGURATION: Update these keys to match your images
# ==========================================

# Input JSONL Keys
QUESTION_ID_KEY = 'question_id'
TEXT_ANSWER_KEY = 'text'
IMAGE_PATH_KEY = 'image'

# Output CSV Multi-Index Headers
OUT_COL_Y_N = 'Y->N'
OUT_COL_N_Y = 'N->Y'
OUT_COL_SCORE = 'Score'
OUT_INDEX_NAME = 'Task'

# ==========================================

def load_jsonl_answers(filepath):
    """
    Reads a JSONL file and extracts answers using the configured keys.
    """
    answers = {}
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return answers
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Use the configurable keys to grab the data
            q_id = str(data.get(QUESTION_ID_KEY, ''))
            image_path = str(data.get(IMAGE_PATH_KEY, ''))
            
            # Clean the answer text
            ans_text = str(data.get(TEXT_ANSWER_KEY, '')).strip().lower().replace('.', '')
            first_word = ans_text.split(' ')[0] if ans_text else ""
            
            # Create a combined string to search for the subtask name
            search_string = f"{q_id} {image_path}".lower()
            
            answers[q_id] = {
                'answer': first_word,
                'search_string': search_string
            }
            
    return answers

def main():
    print("Loading eval results...")
    results_df = pd.read_csv('/teamspace/studios/this_studio/cd_rethink/outputs/mme/mme_eval_results.csv')
    
    baseline_df = results_df[results_df['method'] == 'baseline (greedy)'].copy()
    cd_df = results_df[results_df['method'] != 'baseline (greedy)'].copy()
    
    baseline_files = dict(zip(baseline_df['subtask'], baseline_df['answer_file']))
    
    final_data = []
    
    print("Calculating transitions per subtask...")
    for _, row in cd_df.iterrows():
        subtask = row['subtask']
        subtask_lower = str(subtask).lower()
        method = row['method']
        cd_file = row['answer_file']
        accuracy = row['accuracy']
        
        baseline_file = baseline_files.get(subtask)
        
        if not baseline_file:
            continue
            
        baseline_answers = load_jsonl_answers(baseline_file)
        cd_answers = load_jsonl_answers(cd_file)
        
        yes_to_no = 0
        no_to_yes = 0
        
        for q_id, base_data in baseline_answers.items():
            if subtask_lower in base_data['search_string']:
                if q_id in cd_answers:
                    base_ans = base_data['answer']
                    cd_ans = cd_answers[q_id]['answer']
                    
                    if base_ans == 'yes' and cd_ans == 'no':
                        yes_to_no += 1
                    elif base_ans == 'no' and cd_ans == 'yes':
                        no_to_yes += 1
                    
        final_data.append({
            'subtask': subtask,
            'method': method,
            'yes_to_no': yes_to_no,
            'no_to_yes': no_to_yes,
            'accuracy': accuracy
        })
        
    print("Formatting final table...")
    df = pd.DataFrame(final_data)
    
    pivot_df = df.pivot(
        index='subtask', 
        columns='method', 
        values=['yes_to_no', 'no_to_yes', 'accuracy']
    )
    
    pivot_df = pivot_df.swaplevel(0, 1, axis=1)
    pivot_df = pivot_df.sort_index(axis=1, level=0)
    
    metrics_order = ['yes_to_no', 'no_to_yes', 'accuracy']
    methods = pivot_df.columns.levels[0]
    
    new_columns = pd.MultiIndex.from_product([methods, metrics_order], names=['Method', 'Metric'])
    pivot_df = pivot_df.reindex(columns=new_columns)
    
    # Apply the configured output headers
    pivot_df = pivot_df.rename(columns={
        'yes_to_no': OUT_COL_Y_N, 
        'no_to_yes': OUT_COL_N_Y, 
        'accuracy': OUT_COL_SCORE
    })
    
    pivot_df.index.name = OUT_INDEX_NAME
    
    output_filename = 'mme_final_formatted_table.csv'
    pivot_df.to_csv(output_filename)
    
    print(f"\nDone! Saved properly formatted and filtered data to '{output_filename}'")

if __name__ == "__main__":
    main()