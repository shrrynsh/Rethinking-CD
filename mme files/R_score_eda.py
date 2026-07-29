import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Load and Clean the Data
# ---------------------------------------------------------
def load_and_clean_data(filepath='/teamspace/studios/this_studio/cd_rethink/outputs/mme/mme_no_to_yes_results.csv'):
    df = pd.read_csv(filepath)
    
    # Clean the 'Yes %' column
    df['Yes_pct'] = df['Yes %'].str.rstrip('%').astype(float)
    
    # Clean the 'R_No_To_Yes' column
    def clean_r_score(val):
        if val in ['N/A (Baseline)', 'N/A (No Flips)']:
            return np.nan
        if val == 'Infinity':
            # Cap Infinity at 5.0 for plotting purposes (representing perfect semantic flipping)
            return 5.0 
        return float(val)
    
    df['R_score'] = df['R_No_To_Yes'].apply(clean_r_score)
    
    return df

# ---------------------------------------------------------
# 2. Chunk Subtasks into 3 Groups
# ---------------------------------------------------------
def assign_group(task):
    task = str(task).lower()
    
    # Define keywords for each group
    perception = ['existence', 'count', 'position', 'color']
    recognition = ['ocr', 'artwork', 'celebrity', 'landmark', 'posters', 'scene']
    cognition = ['commonsense', 'numerical', 'translation', 'code']
    
    if any(k in task for k in perception):
        return 'Perception'
    elif any(k in task for k in recognition):
        return 'Recognition'
    elif any(k in task for k in cognition):
        return 'Cognition/Reasoning'
    else:
        return 'Other'

# ---------------------------------------------------------
# 3. Calculate Accuracy Change from Baseline
# ---------------------------------------------------------
def calculate_accuracy_delta(df):
    # Determine if a method is using greedy or sample decoding
    df['Decoding'] = df['Method'].apply(lambda x: 'sample' if 'sample' in x else 'greedy')
    
    # Isolate baselines
    baselines = df[df['Method'].str.contains('baseline')][['Subtask', 'Decoding', 'Accuracy']]
    baselines = baselines.rename(columns={'Accuracy': 'Baseline_Accuracy'})
    
    # Merge baselines back to calculate the delta
    df = df.merge(baselines, on=['Subtask', 'Decoding'], how='left')
    df['Delta_Accuracy'] = df['Accuracy'] - df['Baseline_Accuracy']
    
    return df

# ---------------------------------------------------------
# Main Execution & Plotting
# ---------------------------------------------------------
if __name__ == "__main__":
    # Setup styles
    sns.set_theme(style="whitegrid")
    
    # Data Prep
    df = load_and_clean_data()
    df['Group'] = df['Subtask'].apply(assign_group)
    df = calculate_accuracy_delta(df)
    
    # Filter out baselines and rows with no flips (NaN R_score) for analytical plots
    analysis_df = df[~df['Method'].str.contains('baseline')].dropna(subset=['R_score'])

    # ---------------------------------------------------------
    # Plot 1: Correlation between Delta Accuracy and R Score
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=analysis_df, 
        x='R_score', 
        y='Delta_Accuracy', 
        hue='Group', 
        style='Group',
        s=100, 
        palette='Set1'
    )
    
    # Add reference lines
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.6, label='R=1 (Random Shift)')
    plt.axhline(y=0.0, color='gray', linestyle='--', alpha=0.6, label='No Acc Change')
    
    plt.title('Plot 1: Change in Accuracy vs. Semantic Flip Ratio (R Score)', fontsize=14)
    plt.xlabel('R Score (Cap at 5.0 for Infinity)', fontsize=12)
    plt.ylabel('Change in Accuracy from Baseline', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plot1_acc_vs_rscore.png')
    plt.show()

    # ---------------------------------------------------------
    # Plot 2: R Score for Each Method
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=analysis_df, 
        x='R_score', 
        y='Method', 
        hue='Group',
        palette='Set2'
    )
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.6)
    
    plt.title('Plot 2: Distribution of R Scores by Method', fontsize=14)
    plt.xlabel('Semantic Flip Ratio (R Score)', fontsize=12)
    plt.ylabel('Decoding Method', fontsize=12)
    plt.legend(title='Task Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plot2_rscore_by_method.png')
    plt.show()

    # ---------------------------------------------------------
    # Plot 3: R Score vs. Yes Percentage
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=analysis_df, 
        x='R_score', 
        y='Yes_pct', 
        hue='Group', 
        size='Delta_Accuracy',
        sizes=(20, 200),
        palette='Dark2',
        alpha=0.8
    )
    
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.6)
    
    plt.title('Plot 3: Semantic Flip Ratio (R Score) vs. Overall Yes Percentage', fontsize=14)
    plt.xlabel('R Score', fontsize=12)
    plt.ylabel('Yes Percentage (%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plot3_rscore_vs_yes_pct.png')
    plt.show()

    print("Data processed successfully. Three plots have been generated and saved.")