import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 

import pandas as pd
import seaborn as sns


# 【設定エリア】
TARGET_PARAMS = [
    "Capacity",
    "Lifetime",
    "Topologies",
]

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

REGEX_INTERMEDIATE = re.compile(
    r"Intermediate results.*?"
    r"successRatio=([\d\.]+).*?"
    r"searchTime=([\d\.]+).*?"
    r"successfulTime=([\d\.]+).*?"
    r"unsuccessfulTime=([\d\.]+)",
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
    base_res = {'Method': label, **extracted_params}

    
    final_match = REGEX_FINAL_RESULT.search(content)
    if final_match:
        final_res = {
            **base_res,
            'Success Ratio': float(final_match.group(1)),
            'Search Time': float(final_match.group(2)),
            'Successful Time': float(final_match.group(3)),
            'Failed Time': float(final_match.group(4)),
            'Status': 'Completed'
        }
    else:
        
        int_matches = REGEX_INTERMEDIATE.findall(content)
        if int_matches:
            
            int_data = [
                {
                    'Success Ratio': float(m[0]),
                    'Search Time': float(m[1]),
                    'Successful Time': float(m[2]),
                    'Failed Time': float(m[3])
                }
                for m in int_matches
            ]
            df_int = pd.DataFrame(int_data)
            means = df_int.mean()
            
            final_res = {
                **base_res,
                'Success Ratio': means['Success Ratio'],
                'Search Time': means['Search Time'],
                'Successful Time': means['Successful Time'],
                'Failed Time': means['Failed Time'],
                'Status': f'Partial (N={len(int_matches)})' 
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

    if df_final.empty:
        return

    sns.set(style="whitegrid")
    
    df_plot = df_final.copy()
    df_plot['Success Ratio'] = df_plot['Success Ratio'] * 100

    methods = df_plot['Method'].tolist()
    
    # --- 1. Success Ratio Bar Chart ---
    plt.figure(figsize=(8, 6))
    
    colors_sr = sns.color_palette('viridis', len(df_plot))
    
    bars1 = plt.bar(df_plot['Method'], df_plot['Success Ratio'], color=colors_sr, width=0.2)

    plt.title('Average Success Ratio', fontsize=14)
    plt.ylabel('Success Ratio (%)', fontsize=12) 
    plt.ylim(0, 105) 
    
    handles_sr = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_sr, methods)]
    plt.legend(handles=handles_sr, loc='upper left', title='Method')

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    out_path_sr = output_dir / 'comparison_bar_success_ratio.png'
    plt.tight_layout()
    plt.savefig(out_path_sr)
    plt.close()
    print(f"[Info] Saved bar chart: {out_path_sr}")

    # --- 2. Search Time Bar Chart ---
    plt.figure(figsize=(8, 6))
    
    colors_st = sns.color_palette('magma', len(df_plot))

    bars2 = plt.bar(df_plot['Method'], df_plot['Search Time'], color=colors_st, width=0.2)

    plt.title('Average Search Time', fontsize=14)
    plt.ylabel('Search Time (ms)', fontsize=12)
    
    handles_st = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_st, methods)]
    plt.legend(handles=handles_st, loc='upper left', title='Method')
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    out_path_st = output_dir / 'comparison_bar_search_time.png'
    plt.tight_layout()
    plt.savefig(out_path_st)
    plt.close()
    print(f"[Info] Saved bar chart: {out_path_st}")

def plot_timeseries(df_ts: pd.DataFrame, output_dir: Path):

    if df_ts.empty:
        return

    sns.set(style="whitegrid")
    
    df_plot = df_ts.copy()
    df_plot['Success Ratio'] = df_plot['Success Ratio'] * 100

    # --- 1. Success Ratio Time Series ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Time', y='Success Ratio', hue='Method', data=df_plot, marker='o')
    
    plt.title('Comparison: Success Ratio', fontsize=14)
    plt.ylabel('Success Ratio (%)', fontsize=12)
    plt.xlabel('Time (h)', fontsize=12)
    plt.ylim(0, 105)
    
    plt.legend(loc='upper left', title='Method')
    
    out_path_sr = output_dir / 'comparison_ts_success_ratio.png'
    plt.tight_layout()
    plt.savefig(out_path_sr)
    plt.close()
    print(f"[Info] Saved time series chart: {out_path_sr}")

    # --- 2. Search Time Time Series ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Time', y='Search Time', hue='Method', data=df_plot, marker='o')
    
    plt.title('Comparison: Search Time', fontsize=14)
    plt.ylabel('Search Time (ms)', fontsize=12)
    plt.xlabel('Time (h)', fontsize=12)
    
    plt.legend(loc='upper left', title='Method')
    
    out_path_st = output_dir / 'comparison_ts_search_time.png'
    plt.tight_layout()
    plt.savefig(out_path_st)
    plt.close()
    print(f"[Info] Saved time series chart: {out_path_st}")

def plot_individual_charts(df_ts: pd.DataFrame, label: str, output_dir: Path):

    if df_ts.empty:
        return

    indiv_dir = output_dir / "individual_plots"
    indiv_dir.mkdir(exist_ok=True)
    
    sns.set(style="whitegrid")
    
    df_plot = df_ts.copy()
    df_plot['Success Ratio'] = df_plot['Success Ratio'] * 100

    safe_label = re.sub(r'[\\/*?:"<>|]', "_", label)

    # --- 1. Individual Success Ratio ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(x='Time', y='Success Ratio', data=df_plot, marker='o', color='green')
    
    plt.title(f'[{label}] Success Ratio', fontsize=12)
    plt.ylabel('Success Ratio (%)', fontsize=10)
    plt.xlabel('Time (h)', fontsize=10)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(indiv_dir / f"{safe_label}_success_ratio.png")
    plt.close()

    # --- 2. Individual Search Time ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(x='Time', y='Search Time', data=df_plot, marker='o', color='purple')
    
    plt.title(f'[{label}] Search Time (ms)', fontsize=12)
    plt.ylabel('Search Time (ms)', fontsize=10)
    plt.xlabel('Time (h)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(indiv_dir / f"{safe_label}_search_time.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize and compare network simulation logs.")
    parser.add_argument("files", nargs='*', help="Log files OR directory (default: 'logs').")
    parser.add_argument("--labels", nargs='*', help="Custom labels for the files.")
    parser.add_argument("--out", default="results", help="Base output directory.")

    args = parser.parse_args()

    file_paths = []
    
    if not args.files:
        default_dir = Path("logs")
        if default_dir.exists() and default_dir.is_dir():
            print(f"[Info] No arguments provided. Scanning default directory: {default_dir}")
            file_paths = sorted(list(default_dir.glob("*.txt")))
        else:
            print("[Error] No files provided and 'logs' directory not found.")
            sys.exit(1)
    elif len(args.files) == 1 and Path(args.files[0]).is_dir():
        target_dir = Path(args.files[0])
        print(f"[Info] Scanning directory: {target_dir}")
        file_paths = sorted(list(target_dir.glob("*.txt")))
    else:
        file_paths = [Path(f) for f in args.files]

    labels = args.labels if args.labels else [p.stem for p in file_paths]

    if not file_paths:
        print("[Error] No log files found.")
        sys.exit(1)
    if len(file_paths) != len(labels):
        print(f"[Error] Label count mismatch. Files: {len(file_paths)}, Labels: {len(labels)}")
        sys.exit(1)

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
            plot_individual_charts(df, label, output_dir)

        if final:
            final_results.append(final)

    if not final_results:
        print("[Error] No valid data found in logs.")
        sys.exit(1)

    df_final = pd.DataFrame(final_results)

    # Generate Combined Graphs 
    df_final['Method'] = pd.Categorical(df_final['Method'], categories=labels, ordered=True)
    df_final.sort_values('Method', inplace=True)
    
    plot_comparison(df_final, output_dir)
    
    if all_ts_data:
        df_ts_all = pd.concat(all_ts_data, ignore_index=True)
        df_ts_all['Method'] = pd.Categorical(df_ts_all['Method'], categories=labels, ordered=True)
        df_ts_all.sort_values('Method', inplace=True)
        plot_timeseries(df_ts_all, output_dir)

    # Save CSV
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
    print(f"\n[Done] Check the output directory: {output_dir}")
    print(f"       Individual plots are in: {output_dir / 'individual_plots'}")

if __name__ == "__main__":
    main()