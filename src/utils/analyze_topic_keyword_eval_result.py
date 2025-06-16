import os
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json

def find_csv_files(topic_base_dir: str) -> Dict[str, List[str]]:
    """
    Find all CSV files in eval directories within topic directories.
    
    Returns:
        Dictionary mapping topic_name to list of CSV file paths
    """
    topic_csv_files = {}
    
    for topic_dir in Path(topic_base_dir).iterdir():
        if topic_dir.is_dir():
            topic_name = topic_dir.name
            eval_dir = topic_dir / "eval"
            
            if eval_dir.exists():
                csv_files = []
                # Search for CSV files in eval directory and subdirectories
                for csv_file in eval_dir.rglob("*.csv"):
                    csv_files.append(str(csv_file))
                
                if csv_files:
                    topic_csv_files[topic_name] = csv_files
    
    return topic_csv_files

def read_and_aggregate_results(topic_csv_files: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Read all CSV files and aggregate results into a single DataFrame.
    
    Returns:
        DataFrame with columns: topic_name, method_name, and all evaluation metrics
    """
    all_results = []
    
    for topic_name, csv_files in topic_csv_files.items():
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Add topic information
                df['topic_name'] = topic_name
                df['csv_file'] = os.path.basename(csv_file)
                
                # Rename 'name' column to 'method_name' if it exists
                if 'name' in df.columns:
                    df = df.rename(columns={'name': 'method_name'})
                
                all_results.append(df)
                
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

def find_best_performers(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Find the best performing topic and method for each metric.
    
    Returns:
        Dictionary mapping metric names to best performer info
    """
    # Get all numeric columns (metrics)
    metric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    best_performers = {}
    
    for metric in metric_columns:
        if metric in df.columns:
            # Find the row with maximum value for this metric
            max_idx = df[metric].idxmax()
            best_row = df.loc[max_idx]
            
            best_performers[metric] = {
                'topic_name': best_row['topic_name'],
                'method_name': best_row.get('method_name', 'Unknown'),
                'value': best_row[metric],
                'csv_file': best_row.get('csv_file', 'Unknown')
            }
    
    return best_performers

def generate_summary_report(df: pd.DataFrame, best_performers: Dict, output_path: str):
    """
    Generate a comprehensive summary report.
    """
    with open(output_path, 'w') as f:
        f.write("# Topic Keyword Evaluation Analysis Report\n\n")
        
        # Overview
        f.write("## Overview\n")
        f.write(f"- Total topics analyzed: {df['topic_name'].nunique()}\n")
        f.write(f"- Total evaluation runs: {len(df)}\n")
        f.write(f"- Unique methods: {df.get('method_name', pd.Series()).nunique()}\n\n")
        
        # Best performers for each metric
        f.write("## Best Performers by Metric\n\n")
        for metric, info in best_performers.items():
            f.write(f"### {metric.upper()}\n")
            f.write(f"- **Best Topic**: {info['topic_name']}\n")
            f.write(f"- **Method**: {info['method_name']}\n")
            f.write(f"- **Value**: {info['value']:.4f}\n")
            f.write(f"- **Source**: {info['csv_file']}\n\n")
        
        # Topic-wise summary
        f.write("## Topic-wise Performance Summary\n\n")
        if 'topic_name' in df.columns:
            topic_summary = df.groupby('topic_name').agg({
                col: ['mean', 'max'] for col in df.select_dtypes(include=['float64', 'int64']).columns
            }).round(4)
            
            f.write("### Average Performance by Topic\n")
            f.write("```\n")
            f.write(topic_summary.to_string())
            f.write("\n```\n\n")
        
        # Method-wise summary
        if 'method_name' in df.columns:
            f.write("## Method-wise Performance Summary\n\n")
            method_summary = df.groupby('method_name').agg({
                col: ['mean', 'max', 'count'] for col in df.select_dtypes(include=['float64', 'int64']).columns
            }).round(4)
            
            f.write("### Average Performance by Method\n")
            f.write("```\n")
            f.write(method_summary.to_string())
            f.write("\n```\n\n")

def generate_json_report(best_performers: Dict, output_path: str):
    """
    Generate a JSON report for programmatic access.
    """
    with open(output_path, 'w') as f:
        json.dump(best_performers, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Analyze topic keyword evaluation results")
    parser.add_argument("--topic-base-dir", type=str, required=True,
                       help="Base directory containing topic directories")
    parser.add_argument("--output-dir", type=str, 
                       help="Output directory for reports (defaults to topic-base-dir)")
    
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = args.topic_base_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 Finding CSV files in topic directories...")
    topic_csv_files = find_csv_files(args.topic_base_dir)
    
    if not topic_csv_files:
        print("❌ No CSV files found in any topic directories!")
        return
    
    print(f"📊 Found evaluation results for {len(topic_csv_files)} topics")
    for topic, files in topic_csv_files.items():
        print(f"  - {topic}: {len(files)} files")
    
    print("📖 Reading and aggregating results...")
    df = read_and_aggregate_results(topic_csv_files)
    
    if df.empty:
        print("❌ No valid data found in CSV files!")
        return
    
    print("🏆 Finding best performers...")
    best_performers = find_best_performers(df)
    
    # Generate reports
    summary_path = os.path.join(args.output_dir, "evaluation_summary_report.md")
    json_path = os.path.join(args.output_dir, "best_performers.json")
    csv_path = os.path.join(args.output_dir, "all_results.csv")
    
    print("📝 Generating reports...")
    generate_summary_report(df, best_performers, summary_path)
    generate_json_report(best_performers, json_path)
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Reports generated:")
    print(f"  - Summary: {summary_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Raw data: {csv_path}")
    
    # Print quick summary to console
    print("\n🎯 Quick Summary - Best Performers:")
    for metric, info in list(best_performers.items())[:5]:  # Show first 5 metrics
        print(f"  {metric}: {info['topic_name']} ({info['method_name']}) = {info['value']:.4f}")

if __name__ == "__main__":
    main()
