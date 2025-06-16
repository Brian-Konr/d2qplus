import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def visualize_topic_distribution(topic_dir_path, save_plot=True):
    """
    Visualize topic distribution for a single topic directory
    """
    doc_topics_path = os.path.join(topic_dir_path, "doc_topics.jsonl")
    
    if not os.path.exists(doc_topics_path):
        print(f"Skipping {topic_dir_path}: doc_topics.jsonl not found")
        return None
    
    # Load topic data
    with open(doc_topics_path, "r") as f:
        corpus_topic = [json.loads(line) for line in f]
    
    # Calculate topic numbers per document
    topic_num = []
    for doc in corpus_topic:
        topic_num.append(len(doc['topics']))
    
    # Calculate statistics
    avg_topics = sum(topic_num) / len(topic_num) if topic_num else 0
    max_topics = max(topic_num) if topic_num else 0
    min_topics = min(topic_num) if topic_num else 0
    
    stats = {
        'avg_topics': avg_topics,
        'max_topics': max_topics,
        'min_topics': min_topics,
        'total_docs': len(topic_num)
    }
    
    print(f"\n=== {os.path.basename(topic_dir_path)} ===")
    print(f"Average number of topics per document: {avg_topics:.2f}")
    print(f"Max number of topics per document: {max_topics}")
    print(f"Min number of topics per document: {min_topics}")
    print(f"Total documents: {len(topic_num)}")
    
    if save_plot:
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Create subplot layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram with KDE
        sns.histplot(topic_num, bins=30, kde=True, ax=ax1)
        ax1.set_title(f'Topic Distribution - {os.path.basename(topic_dir_path)}')
        ax1.set_xlabel('Number of Topics per Document')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(0, max_topics + 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Box plot for additional insights
        sns.boxplot(y=topic_num, ax=ax2)
        ax2.set_title('Topic Distribution Box Plot')
        ax2.set_ylabel('Number of Topics per Document')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f'Avg: {avg_topics:.2f}\nMax: {max_topics}\nMin: {min_topics}\nDocs: {len(topic_num)}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(topic_dir_path, "topic_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot to: {plot_path}")
    
    return stats

def process_all_topic_directories(base_dir):
    """
    Process all topic directories and create visualizations
    """
    all_stats = []
    
    # Find all subdirectories that contain doc_topics.jsonl
    try:
        for root, dirs, files in os.walk(base_dir):
            try:
                if "doc_topics.jsonl" in files:
                    stats = visualize_topic_distribution(root, save_plot=True)
                    if stats:
                        stats['config_dir'] = os.path.basename(root)
                        stats['full_path'] = root
                        all_stats.append(stats)
            except PermissionError:
                print(f"Permission denied: Skipping {root}")
                continue
            except Exception as e:
                print(f"Error processing {root}: {e}")
                continue
    except PermissionError:
        print(f"Permission denied: Cannot access base directory {base_dir}")
        return None
    except Exception as e:
        print(f"Error walking directory {base_dir}: {e}")
        return None
    
    # Create summary DataFrame
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_df = summary_df.sort_values('avg_topics', ascending=False)
        
        # Save summary
        summary_path = os.path.join(base_dir, "topic_distribution_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        return summary_df
    else:
        print("No topic directories found!")
        return None

def process_topic_info_dataframes(base_dir):
    """
    Process topic_info_dataframe.pkl files in all topic directories
    """
    topic_info_summary = []
    
    for root, dirs, files in os.walk(base_dir):
        pkl_path = os.path.join(root, "topic_info_dataframe.pkl")
        csv_path = os.path.join(root, "topic_info_dataframe.csv")
        
        if os.path.exists(pkl_path):
            try:
                # Load pickle file
                topic_info_df = pd.read_pickle(pkl_path)
                
                # Save as CSV if not exists
                if not os.path.exists(csv_path):
                    topic_info_df.to_csv(csv_path, index=False)
                    print(f"Saved CSV: {csv_path}")
                
                # Collect summary statistics
                config_name = os.path.basename(root)
                num_topics = len(topic_info_df) - 1  # Exclude outlier topic (-1)
                
                summary_info = {
                    'config_dir': config_name,
                    'num_topics': num_topics,
                    'total_docs_in_topics': topic_info_df['Count'].sum() if 'Count' in topic_info_df.columns else 0,
                    'avg_docs_per_topic': topic_info_df['Count'].mean() if 'Count' in topic_info_df.columns else 0,
                    'max_docs_per_topic': topic_info_df['Count'].max() if 'Count' in topic_info_df.columns else 0,
                    'min_docs_per_topic': topic_info_df['Count'].min() if 'Count' in topic_info_df.columns else 0,
                }
                
                topic_info_summary.append(summary_info)
                
            except Exception as e:
                print(f"Error processing {pkl_path}: {e}")
    
    # Create and save summary
    if topic_info_summary:
        topic_summary_df = pd.DataFrame(topic_info_summary)
        topic_summary_df = topic_summary_df.sort_values('num_topics', ascending=False)
        
        summary_path = os.path.join(base_dir, "topic_info_summary.csv")
        topic_summary_df.to_csv(summary_path, index=False)
        print(f"Saved topic info summary to: {summary_path}")
        
        return topic_summary_df
    else:
        print("No topic info files found!")
        return None

# Generate a comprehensive report combining all analyses
def generate_comprehensive_report(base_dir, distribution_summary, topic_info_summary):
    """
    Generate a comprehensive report combining all analyses
    """
    report_lines = []
    report_lines.append("# Topic Modeling Grid Search Analysis Report")
    report_lines.append(f"Generated from: {base_dir}")
    report_lines.append(f"Total configurations analyzed: {len(distribution_summary) if distribution_summary is not None else 0}")
    report_lines.append("")
    
    if distribution_summary is not None and topic_info_summary is not None:
        # Merge the two summaries
        merged_summary = distribution_summary.merge(
            topic_info_summary[['config_dir', 'num_topics', 'avg_docs_per_topic']], 
            on='config_dir', 
            how='left'
        )
        
        report_lines.append("## Top Configurations by Average Topics per Document")
        top_avg = merged_summary.nlargest(5, 'avg_topics')
        for _, row in top_avg.iterrows():
            report_lines.append(f"- **{row['config_dir']}**: {row['avg_topics']:.2f} avg topics, {row['num_topics']} total topics")
        report_lines.append("")
        
        report_lines.append("## Top Configurations by Total Number of Topics")
        top_num = merged_summary.nlargest(5, 'num_topics')
        for _, row in top_num.iterrows():
            report_lines.append(f"- **{row['config_dir']}**: {row['num_topics']} topics, {row['avg_topics']:.2f} avg per doc")
        report_lines.append("")
        
        report_lines.append("## Configuration Analysis")
        report_lines.append(f"- Average topics per document range: {merged_summary['avg_topics'].min():.2f} - {merged_summary['avg_topics'].max():.2f}")
        report_lines.append(f"- Total topics range: {merged_summary['num_topics'].min()} - {merged_summary['num_topics'].max()}")
        report_lines.append(f"- Most documents in single config: {merged_summary['total_docs'].max()}")
        report_lines.append("")
        
        # Save detailed summary
        detailed_summary_path = os.path.join(base_dir, "detailed_analysis_summary.csv")
        merged_summary.to_csv(detailed_summary_path, index=False)
        report_lines.append(f"Detailed analysis saved to: {detailed_summary_path}")
    
    # Save report
    report_path = os.path.join(base_dir, "analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Comprehensive report saved to: {report_path}")
    print("\n" + "\n".join(report_lines))

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize topic modeling results")
    parser.add_argument(
        "--topic_base_dir", 
        type=str, 
        help="Base directory containing topic modeling results"
    )
    args = parser.parse_args()
    
    topic_base_dir = args.topic_base_dir
    
    if not os.path.exists(topic_base_dir):
        print(f"Error: Directory {topic_base_dir} does not exist!")
        return
    
    print(f"Processing topic modeling results from: {topic_base_dir}")
    
    # Process all directories
    summary_stats = process_all_topic_directories(topic_base_dir)

    # Create overall comparison visualization
    if summary_stats is not None and len(summary_stats) > 0:
        plt.figure(figsize=(15, 10))
        
        # Create subplots for comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. Average topics comparison
        summary_stats_sorted = summary_stats.sort_values('avg_topics')
        ax1.barh(range(len(summary_stats_sorted)), summary_stats_sorted['avg_topics'])
        ax1.set_yticks(range(len(summary_stats_sorted)))
        ax1.set_yticklabels(summary_stats_sorted['config_dir'], fontsize=8)
        ax1.set_xlabel('Average Topics per Document')
        ax1.set_title('Average Topics per Document by Configuration')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Max topics comparison
        summary_stats_sorted = summary_stats.sort_values('max_topics')
        ax2.barh(range(len(summary_stats_sorted)), summary_stats_sorted['max_topics'])
        ax2.set_yticks(range(len(summary_stats_sorted)))
        ax2.set_yticklabels(summary_stats_sorted['config_dir'], fontsize=8)
        ax2.set_xlabel('Maximum Topics per Document')
        ax2.set_title('Maximum Topics per Document by Configuration')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Scatter plot: avg vs max topics
        ax3.scatter(summary_stats['avg_topics'], summary_stats['max_topics'], alpha=0.7)
        ax3.set_xlabel('Average Topics per Document')
        ax3.set_ylabel('Maximum Topics per Document')
        ax3.set_title('Average vs Maximum Topics per Document')
        ax3.grid(alpha=0.3)
        
        # Add labels for outliers
        for i, row in summary_stats.iterrows():
            if row['max_topics'] > summary_stats['max_topics'].quantile(0.8):
                ax3.annotate(row['config_dir'], 
                            (row['avg_topics'], row['max_topics']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.7)
        
        # 4. Total documents comparison
        ax4.bar(range(len(summary_stats)), summary_stats['total_docs'])
        ax4.set_xticks(range(len(summary_stats)))
        ax4.set_xticklabels(summary_stats['config_dir'], rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Total Documents')
        ax4.set_title('Total Documents by Configuration')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_plot_path = os.path.join(topic_base_dir, "topic_distribution_comparison.png")
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot to: {comparison_plot_path}")

    # Process all topic info dataframes
    topic_summary = process_topic_info_dataframes(topic_base_dir)

    # Generate comprehensive report
    if summary_stats is not None and topic_summary is not None:
        generate_comprehensive_report(topic_base_dir, summary_stats, topic_summary)

if __name__ == "__main__":
    main()