import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Configuration: Regex Patterns ---
# Modify these patterns if the log format changes.
REGEX_TIME_SERIES = re.compile(
    r"time=(\d+).*?"
    r"successRatio=([\d\.]+).*?"
    r"searchTime=([\d\.]+).*?"
    r"successfulTime=([\d\.]+).*?"
    r"unsuccessfulTime=([\d\.]+)"
)

REGEX_FINAL_RESULT = re.compile(
    r"Final results.*?"
    r"avgSuccessRatio=([\d\.]+).*?"
    r"avgSearchTime=([\d\.]+).*?"
    r"avgSuccessfulTime=([\d\.]+).*?"
    r"avgFailedTime=([\d\.]+)"
)

def parse_log_file(file_path: Path, label: str) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Parses a single log file to extract time-series data and final results.

    Args:
        file_path (Path): Path to the log file.
        label (str): Label for the dataset (e.g., method name).

    Returns:
        Tuple[pd.DataFrame, Optional[Dict]]: 
            - DataFrame containing time-series data.
            - Dictionary containing final results (or None if extraction fails).
    """
    if not file_path.exists():
        print(f"[Warning] File not found: {file_path}. Skipping.")
        return pd.DataFrame(), None

    try:
        # Try reading with utf-8, fallback to cp932 (Windows)
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = file_path.read_text(encoding='cp932')
    except Exception as e:
        print(f"[Error] Could not read file {file_path}: {e}")
        return pd.DataFrame(), None

    # 1. Extract Time Series Data
    ts_matches = REGEX_TIME_SERIES.findall(content)
    ts_data = [
        {
            'Method': label,
            'Time': int(m[0]),
            'Success Ratio': float(m[1]),
            'Search Time': float(m[2]),
            'Successful Time': float(m[3]),
            'Failed Time': float(m[4])
        }
        for m in ts_matches
    ]
    df_ts = pd.DataFrame(ts_data)

    # 2. Extract Final Results
    final_res = None
    final_match = REGEX_FINAL_RESULT.search(content)

    if final_match:
        final_res = {
            'Method': label,
            'Success Ratio': float(final_match.group(1)),
            'Search Time': float(final_match.group(2)),
            'Successful Time': float(final_match.group(3)),
            'Failed Time': float(final_match.group(4)),
            'Status': 'Completed'
        }
    elif not df_ts.empty:
        # Fallback: Use the last record of time series if final result is missing
        last_row = df_ts.iloc[-1]
        final_res = {
            'Method': label,
            'Success Ratio': last_row['Success Ratio'],
            'Search Time': last_row['Search Time'],
            'Successful Time': last_row['Successful Time'],
            'Failed Time': last_row['Failed Time'],
            'Status': f'Running (Time={last_row["Time"]})'
        }

    return df_ts, final_res


def plot_comparison(df_final: pd.DataFrame, output_dir: Path):
    """Generates and saves bar charts for final result comparison."""
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot: Success Ratio
    sns.barplot(x='Method', y='Success Ratio', data=df_final, ax=axes[0], palette='viridis', hue='Method')
    axes[0].set_title('Average Success Ratio (Higher is Better)', fontsize=14)
    axes[0].set_ylim(0, 1.05)
    for i, row in df_final.iterrows():
        axes[0].text(i, row['Success Ratio'] + 0.02, f"{row['Success Ratio']:.3f}", ha='center', fontweight='bold')

    # Plot: Search Time
    sns.barplot(x='Method', y='Search Time', data=df_final, ax=axes[1], palette='magma', hue='Method')
    axes[1].set_title('Average Search Time (ms) (Lower is Better)', fontsize=14)
    for i, row in df_final.iterrows():
        axes[1].text(i, row['Search Time'] + 50, f"{int(row['Search Time'])}", ha='center', fontweight='bold')

    output_path = output_dir / 'comparison_bar_chart.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[Info] Saved bar chart: {output_path}")


def plot_timeseries(df_ts: pd.DataFrame, output_dir: Path):
    """Generates and saves line charts for time-series analysis."""
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot: Success Ratio over Time
    sns.lineplot(x='Time', y='Success Ratio', hue='Method', data=df_ts, ax=axes[0], marker='o')
    axes[0].set_title('Success Ratio over Time', fontsize=14)

    # Plot: Search Time over Time
    sns.lineplot(x='Time', y='Search Time', hue='Method', data=df_ts, ax=axes[1], marker='o')
    axes[1].set_title('Search Time over Time (ms)', fontsize=14)

    output_path = output_dir / 'comparison_time_series.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[Info] Saved time series chart: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize and compare network simulation logs.")
    parser.add_argument("files", nargs='+', help="List of log files to process.")
    parser.add_argument("--labels", nargs='*', help="Custom labels for the files (space-separated).")
    parser.add_argument("--out", default=".", help="Output directory for images and CSV.")

    args = parser.parse_args()

    # Prepare file paths and labels
    file_paths = [Path(f) for f in args.files]
    labels = args.labels if args.labels else [p.stem for p in file_paths]

    if len(file_paths) != len(labels):
        print("[Error] The number of labels must match the number of files.")
        sys.exit(1)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Processing {len(file_paths)} files ---")

    all_ts_data = []
    final_results = []

    for path, label in zip(file_paths, labels):
        print(f"Reading: {label} ({path})")
        df, final = parse_log_file(path, label)
        
        if not df.empty:
            all_ts_data.append(df)
        if final:
            final_results.append(final)

    if not final_results:
        print("[Error] No valid data found in the provided logs.")
        sys.exit(1)

    # Aggregation
    df_final = pd.DataFrame(final_results)
    
    # Generate Outputs
    plot_comparison(df_final, output_dir)

    if all_ts_data:
        df_ts_all = pd.concat(all_ts_data, ignore_index=True)
        plot_timeseries(df_ts_all, output_dir)

    # Save Summary CSV
    csv_path = output_dir / 'analysis_summary.csv'
    df_final.to_csv(csv_path, index=False)
    print(f"[Info] Saved summary CSV: {csv_path}")

    print("\n=== Analysis Summary ===")
    print(df_final[['Method', 'Status', 'Success Ratio', 'Search Time']].to_string(index=False))


if __name__ == "__main__":
    main()