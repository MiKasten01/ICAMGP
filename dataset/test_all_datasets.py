import json
import os
import sys
import traceback
import pandas as pd
import numpy as np

# Add parent directory to sys.path to allow importing dataset package if running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dataset.new_dataset import load_dataset

def test_all_datasets():
    # Determine config path relative to this script
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_config.json')
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

    print(f"Found {len(config)} datasets in config.")
    
    results = []
    success_count = 0
    fail_count = 0

    for dataset_name in config.keys():
        print(f"Processing {dataset_name}...")
        try:
            ds = load_dataset(dataset_name, config_path=config_path)
            # Ensure stats are calculated
            if not hasattr(ds, 'stat') or not ds.stat:
                ds._calculate_stats()
            
            stat = ds.stat
            
            # Calculate scalar means for X if they are arrays
            x_mean_val = stat.get('X_mean')
            if isinstance(x_mean_val, np.ndarray):
                x_mean_val = np.mean(x_mean_val)
                
            x_std_val = stat.get('X_std')
            if isinstance(x_std_val, np.ndarray):
                x_std_val = np.mean(x_std_val)

            row = {
                'Dataset': dataset_name,
                'Status': 'SUCCESS',
                'Num': stat.get('Num'),
                'SNPs': stat.get('SNPs'),
                'X_mean': round(float(x_mean_val), 4) if x_mean_val is not None else None,
                'X_std': round(float(x_std_val), 4) if x_std_val is not None else None,
                'Y_mean': round(stat.get('Y_mean'), 4) if stat.get('Y_mean') is not None else None,
                'Y_std': round(stat.get('Y_std'), 4) if stat.get('Y_std') is not None else None,
                'Y_var': round(stat.get('Y_var'), 4) if stat.get('Y_var') is not None else None,
                'h2': round(stat.get('h2'), 4) if stat.get('h2') is not None else None
            }
            results.append(row)
            success_count += 1
            print(f"  -> Success. Num={row['Num']}, SNPs={row['SNPs']}")
            
        except Exception as e:
            fail_count += 1
            print(f"  -> Failed: {e}")
            # traceback.print_exc()
            results.append({
                'Dataset': dataset_name,
                'Status': 'FAILED',
                'Error': str(e)
            })

    print("-" * 60)
    print(f"Test Complete. Success: {success_count}, Failed: {fail_count}")
    
    # Save to CSV
    df = pd.DataFrame(results)
    # Reorder columns
    cols = ['Dataset', 'Status', 'Num', 'SNPs', 'X_mean', 'X_std', 'Y_mean', 'Y_std', 'Y_var', 'h2', 'Error']
    # Only keep columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_summary.csv')
    df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")
    print(df)

if __name__ == "__main__":
    test_all_datasets()
