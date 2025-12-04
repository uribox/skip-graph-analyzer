import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# --- Matplotlib Backend Fix (Windowsエラー回避) ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

# ==============================================================================
# 【設定エリア】
# ログから抽出してCSVに残したいパラメータ名
# ==============================================================================
TARGET_PARAMS = [
    "Capacity",
    "Lifetime",
    "Topologies",
]
# ==============================================================================

# --- Regex Patterns ---
REGEX_TIME_SERIES = re.compile(
    r"time=(\d+).*?"
    r"successRatio=([\d\.]+).*?"
    r"searchTime=([\d\.]+).*?"
    r"successfulTime=([\d\.]+).*?"
    r"unsuccessfulTime=([\d\.]+)",
    re.IGNORECASE
)

REGEX_FINAL_RESULT = re.compile(
    r"Final results.*?"
    r"avgSuccessRatio=([\d\.]+).*?"
    r"avgSearchTime=([\d\.]+).*?"
    r"avgSuccessfulTime=([\d\.]+).*?"
    r"avgFailedTime=([\d\.]+)",
    re.IGNORECASE
)

def extract_parameters(content: str) -> Dict[str, str]:
    params = {}
    for key in TARGET_PARAMS:
        pattern = re.compile(rf"{re.escape(key)}\s*[=:]?\s*([\w\-\.]+)", re.IGNORECASE)
        match = pattern.search(content)
        if match:
            params[key] = match.group(1)
        else:
            params[key] = "N/A"
    return params

def parse_log_file(file_path: Path, label: str) -> Tuple[pd.DataFrame, Optional[Dict]]:
    if not file_path.exists():
        print(f"[Warning] File not found: {file_path}. Skipping.")
        return pd.DataFrame(), None

    try:
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = file_path.read_text(encoding='cp932')
    except Exception as e:
        print(f"[Error] Could not read file {file_path}: {e}")
        return pd.DataFrame(), None

    extracted_params = extract_parameters(content)
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

    final_res = None
    final_match = REGEX_FINAL_RESULT.search(content)
    base_res = {'Method': label, **extracted_params}

    if final_match:
        final_res = {
            **base_res,
            'Success Ratio': float(final_match.group(1)),
            'Search Time': float(final_match.group(2)),
            'Successful Time': float(final_match.group(3)),
            'Failed Time': float(final_match.group(4)),
            'Status': 'Completed'
        }
    elif not df_ts.empty:
        last_row = df_ts.iloc[-1]
        final_res = {
            **base_res,
            'Success Ratio': last_row['Success Ratio'],
            'Search Time': last_row['Search Time'],
            'Successful Time': last_row['Successful Time'],
            'Failed Time': last_row['Failed Time'],
            'Status': f'Running (Time={last_row["Time"]})'
        }
    else:
        final_res = {
            **base_res,
            'Success Ratio': 0.0,
            'Search Time': 0.0,
            'Status': 'No Data'
        }

    return df_ts, final_res

def plot_comparison(df_final: pd.DataFrame, output_dir: Path):
    if df_final.empty or df_final['Success Ratio'].sum() == 0:
        return

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(x='Method', y='Success Ratio', data=df_final, ax=axes[0], palette='viridis', hue='Method')
    axes[0].set_title('Average Success Ratio (Higher is Better)', fontsize=14)
    axes[0].set_ylim(0, 1.05)
    for i, row in df_final.iterrows():
        if row['Success Ratio'] > 0:
            axes[0].text(i, row['Success Ratio'] + 0.01, f"{row['Success Ratio']:.3f}", ha='center', fontweight='bold')

    sns.barplot(x='Method', y='Search Time', data=df_final, ax=axes[1], palette='magma', hue='Method')
    axes[1].set_title('Average Search Time (ms) (Lower is Better)', fontsize=14)
    for i, row in df_final.iterrows():
        if row['Search Time'] > 0:
            axes[1].text(i, row['Search Time'] + 50, f"{int(row['Search Time'])}", ha='center', fontweight='bold')

    output_path = output_dir / 'comparison_bar_chart.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[Info] Saved bar chart: {output_path}")

def plot_timeseries(df_ts: pd.DataFrame, output_dir: Path):
    if df_ts.empty:
        return

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.lineplot(x='Time', y='Success Ratio', hue='Method', data=df_ts, ax=axes[0], marker='o')
    axes[0].set_title('Success Ratio over Time', fontsize=14)

    sns.lineplot(x='Time', y='Search Time', hue='Method', data=df_ts, ax=axes[1], marker='o')
    axes[1].set_title('Search Time over Time (ms)', fontsize=14)

    output_path = output_dir / 'comparison_time_series.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[Info] Saved time series chart: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize and compare network simulation logs.")
    parser.add_argument("files", nargs='+', help="List of log files to process.")
    parser.add_argument("--labels", nargs='*', help="Custom labels for the files (must match file count).")
    parser.add_argument("--out", default="results", help="Base output directory.")

    args = parser.parse_args()

    # --- File Collection ---
    file_paths = [Path(f) for f in args.files]
    labels = args.labels if args.labels else [p.stem for p in file_paths]

    if len(file_paths) != len(labels):
        print(f"[Error] Label count mismatch. Files: {len(file_paths)}, Labels: {len(labels)}")
        sys.exit(1)

    # --- Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Processing {len(file_paths)} files ---")
    print(f"[Info] Output Directory: {output_dir}")
    
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
        print("[Error] No valid data found in logs.")
        sys.exit(1)

    df_final = pd.DataFrame(final_results)

    # --- Generate Graphs ---
    # 入力順序（labelsの順序）を守るためにCategorical型を使用
    df_final['Method'] = pd.Categorical(df_final['Method'], categories=labels, ordered=True)
    df_final.sort_values('Method', inplace=True)
    
    plot_comparison(df_final, output_dir)
    
    if all_ts_data:
        df_ts_all = pd.concat(all_ts_data, ignore_index=True)
        # 時系列データも同じ順序を守る
        df_ts_all['Method'] = pd.Categorical(df_ts_all['Method'], categories=labels, ordered=True)
        df_ts_all.sort_values('Method', inplace=True)
        
        plot_timeseries(df_ts_all, output_dir)

    # --- Save CSV ---
    csv_path = output_dir / 'analysis_summary.csv'
    base_cols = ['Method'] + TARGET_PARAMS
    metric_cols = [c for c in df_final.columns if c not in base_cols]
    final_cols = [c for c in base_cols + metric_cols if c in df_final.columns]
    
    df_final[final_cols].to_csv(csv_path, index=False)
    print(f"[Info] Saved summary CSV: {csv_path}")
    
    # Preview
    preview_cols = ['Method', 'Success Ratio', 'Search Time', 'Status']
    print("\n=== Analysis Summary ===")
    print(df_final[[c for c in preview_cols if c in df_final.columns]].to_string(index=False))

if __name__ == "__main__":
    main()