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

# Define a consistent color map for models
all_files = glob.glob("data/grades/topic_grades_*.csv") + glob.glob("data/articulation/articulation_results_*.csv") + glob.glob("data/faithfulness/topic_faithfulness_*.csv")
all_models = set()
for f in all_files:
    model_name_parts = os.path.basename(f).replace('topic_grades_', '').replace('articulation_results_', '').replace('topic_faithfulness_', '').replace('.csv', '').split('_')
    all_models.add('_'.join(model_name_parts[:2]))

colors = plt.cm.get_cmap('tab10', len(all_models))
MODEL_COLORS = {model: colors(i) for i, model in enumerate(sorted(list(all_models)))}


def plot_metrics(grades_df, file_prefix=""):
    if grades_df.empty:
        return

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        pivot_df = grades_df.pivot(index='category', columns='model', values=metric)
        
        n_models = len(pivot_df.columns)
        n_categories = len(pivot_df.index)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bar_width = 0.8 / n_models
        index = np.arange(n_categories)
        
        for i, model in enumerate(pivot_df.columns):
            ax.bar(index + i * bar_width, pivot_df[model], bar_width, label=model, color=MODEL_COLORS.get(model))

        ax.set_xlabel('Category')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} by Category and Model')
        ax.set_xticks(index + bar_width * (n_models - 1) / 2)
        ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"results/{file_prefix}{metric}_by_category.png")
        plt.close()

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

    # Create a directory for results if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Separate sentence topics
    sentence_grades_df = grades_df[grades_df['category'].str.startswith('sentence')]
    other_grades_df = grades_df[~grades_df['category'].str.startswith('sentence')]

    # Plot metrics for both dataframes
    plot_metrics(other_grades_df)
    plot_metrics(sentence_grades_df, file_prefix="sentence_")

def make_articulation_charts():
    # Find all articulation results files
    files = glob.glob("data/articulation/articulation_results_*.csv")
    
    if not files:
        print("No articulation results CSV files found in data/ directory.")
        return

    all_articulation_data = []
    for f in files:
        df = pd.read_csv(f)
        model_name_parts = os.path.basename(f).replace('articulation_results_', '').replace('.csv', '').split('_')
        model_name = '_'.join(model_name_parts[:2])
        
        # Calculate accuracy for boolean-like columns
        for col in ['understands_mcq', 'understands_mcq_none', 'ff_mentions_topic', 'ff_mentions_topic_fake_hint']:
            # The values are strings 'True'/'False', so we compare to 'True'
            if col in df.columns:
                accuracy = (df[col].astype(str) == 'True').mean()
                all_articulation_data.append({'model': model_name, 'metric': col, 'accuracy': accuracy})
            else:
                print(f"Warning: Column '{col}' not found in {f}. Skipping.")
    
    articulation_df = pd.DataFrame(all_articulation_data)

    # Create a directory for results if it doesn't exist
    os.makedirs("results", exist_ok=True)

    pivot_df = articulation_df.pivot(index='metric', columns='model', values='accuracy')
    
    n_models = len(pivot_df.columns)
    n_metrics = len(pivot_df.index)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.8 / n_models
    index = np.arange(n_metrics)
    
    for i, model in enumerate(pivot_df.columns):
        ax.bar(index + i * bar_width, pivot_df[model], bar_width, label=model, color=MODEL_COLORS.get(model))

    ax.set_xlabel('Articulation Metric')
    ax.set_ylabel('Accuracy')
    ax.set_title('Articulation Accuracy by Model')
    ax.set_xticks(index + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in pivot_df.index], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    
    # Add a legend for the models
    model_legend = ax.legend(title='Models')
    ax.add_artist(model_legend)

    # Add a second legend for the abbreviations
    from matplotlib.patches import Patch
    abbreviation_legend_handles = [
        Patch(color='none', label='FF = Freeform Response'),
        Patch(color='none', label='MCQ = Multiple Choice Question')
    ]
    ax.legend(handles=abbreviation_legend_handles, loc='lower right', title='Abbreviations')
    
    plt.tight_layout()
    plt.savefig("results/articulation_accuracy_by_model.png")
    plt.close()

def make_faithfulness_charts():
    # Find all faithfulness results files
    files = glob.glob("data/faithfulness/topic_faithfulness_*.csv")
    
    if not files:
        print("No faithfulness results CSV files found in data/faithfulness/ directory.")
        return

    all_faithfulness_data = []
    for f in files:
        df = pd.read_csv(f)
        model_name_parts = os.path.basename(f).replace('topic_faithfulness_', '').replace('.csv', '').split('_')
        model_name = '_'.join(model_name_parts[:2])
        
        # Calculate faithfulness percentage
        # The values are booleans, so mean will give the percentage of True
        faithfulness_percentage = df['is_faithful'].mean()
        
        all_faithfulness_data.append({'model': model_name, 'faithfulness': faithfulness_percentage})

    if not all_faithfulness_data:
        return
        
    faithfulness_df = pd.DataFrame(all_faithfulness_data)
    faithfulness_df = faithfulness_df.sort_values(by='model')

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(faithfulness_df['model'], faithfulness_df['faithfulness'], color=[MODEL_COLORS.get(m) for m in faithfulness_df['model']])

    ax.set_xlabel('Model')
    ax.set_ylabel('Faithfulness Score')
    ax.set_title('Faithfulness Score by Model')
    ax.set_xticklabels(faithfulness_df['model'], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("results/faithfulness_by_model.png")
    plt.close()

if __name__ == "__main__":
    make_charts()
    make_articulation_charts()
    make_faithfulness_charts()