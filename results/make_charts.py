# results/make_charts.py
"""
In data/topic_grades.csv, we have the performance metrics of the LLM-based category classifier across various topics.
This script reads that CSV file and generates visual charts to illustrate the accuracy, precision, recall, and F1 score for each category.
It saves the charts as PNG files in the results/ directory.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

def make_charts():
    # Find all topic grades files
    files = glob.glob("data/grades/topic_grades_*.csv")
    
    if not files:
        print("No topic grade CSV files found in data/ directory.")
        return

    all_grades = []
    for f in files:
        df = pd.read_csv(f)
        model_name_parts = os.path.basename(f).replace('topic_grades_', '').replace('.csv', '').split('_')
        df['model'] = '_'.join(model_name_parts[:2])
        all_grades.append(df)
    
    grades_df = pd.concat(all_grades, ignore_index=True)

    # Assign colors to models
    models = grades_df['model'].unique()
    colors = plt.cm.get_cmap('tab10', len(models))
    model_colors = {model: colors(i) for i, model in enumerate(models)}

    # Create a directory for results if it doesn't exist
    os.makedirs("results", exist_ok=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        pivot_df = grades_df.pivot(index='category', columns='model', values=metric)
        
        n_models = len(pivot_df.columns)
        n_categories = len(pivot_df.index)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bar_width = 0.8 / n_models
        index = np.arange(n_categories)
        
        for i, model in enumerate(pivot_df.columns):
            ax.bar(index + i * bar_width, pivot_df[model], bar_width, label=model, color=model_colors[model])

        ax.set_xlabel('Category')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} by Category and Model')
        ax.set_xticks(index + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"results/{metric}_by_category.png")
        plt.close()

if __name__ == "__main__":
    make_charts()