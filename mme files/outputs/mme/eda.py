import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_subtask_transitions():
    # 1. Load the multi-level pivot table
    # header=[0, 1] reads the top two rows as the MultiIndex (Method, Metric)
    # index_col=0 sets the Task column as the index
    print("Loading formatted data...")
    df = pd.read_csv('/teamspace/studios/this_studio/cd_rethink/outputs/mme/mme_final_formatted_table.csv', header=[0, 1], index_col=0)

    # 2. Identify all methods that use a 'sample' strategy
    all_methods = df.columns.levels[0]
    sample_methods = [method for method in all_methods if 'sample' in method.lower()]

    # 3. Get the list of subtasks
    subtasks = df.index.dropna().tolist()
    n_tasks = len(subtasks)

    # 4. Set up a grid of subplots (e.g., 4 columns wide)
    cols = 4
    rows = math.ceil(n_tasks / cols)
    
    # Make the figure large enough to comfortably fit all tasks
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    print(f"Generating plots for {n_tasks} subtasks...")
    
    for i, task in enumerate(subtasks):
        ax = axes[i]
        
        y_n_vals = []
        n_y_vals = []
        valid_methods = []
        
        # Extract the metrics for each sample method on this specific task
        for method in sample_methods:
            if method in df.columns.get_level_values(0):
                # Using .get() or checking existence handles any NaN/missing data safely
                y_n = df.loc[task, (method, 'Y->N')]
                n_y = df.loc[task, (method, 'N->Y')]
                
                y_n_vals.append(0 if pd.isna(y_n) else y_n)
                n_y_vals.append(0 if pd.isna(n_y) else n_y)
                valid_methods.append(method)
                
        # 5. Plot the side-by-side bars for this subtask
        x = np.arange(len(valid_methods))
        width = 0.35
        
        ax.bar(x - width/2, y_n_vals, width, label='Y->N (Turned Negative)', color='skyblue', edgecolor='black')
        ax.bar(x + width/2, n_y_vals, width, label='N->Y (Turned Positive)', color='lightcoral', edgecolor='black')
        
        # Formatting the individual plot
        ax.set_title(f'{str(task).upper()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_methods, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Only add the legend to the very first plot to avoid clutter
        if i == 0:
            ax.legend(loc='upper left')

    # 6. Hide any extra, unused subplots in the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 7. Final layout adjustments and save
    fig.tight_layout()
    output_img = 'sample_interventions_subtask_wise.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    
    print(f"Done! Subtask-wise transition grid saved to '{output_img}'")

if __name__ == "__main__":
    plot_subtask_transitions()