## GALA/scripts/extract_features.py

"""Extract power and area features for GALA."""

import os
import argparse
from tqdm import tqdm
import sys
sys.path.append('..')

from utils.power_area_extractor import (
    extract_power_primetime, estimate_power,
    extract_area_yosys, estimate_area
)


def main():
    parser = argparse.ArgumentParser(description='Extract power and area features')
    parser.add_argument('--benchmarks', type=str, required=True,
                       help='Directory containing locked netlists')
    parser.add_argument('--output', type=str, default='features',
                       help='Output directory for features')
    parser.add_argument('--tool', type=str, default='estimate',
                       choices=['primetime', 'estimate'],
                       help='Tool to use for extraction')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Find all Verilog files
    verilog_files = []
    for root, dirs, files in os.walk(args.benchmarks):
        for file in files:
            if file.endswith('.v'):
                verilog_files.append(os.path.join(root, file))
    
    print(f"Found {len(verilog_files)} Verilog files")
    
    # Extract features
    for vfile in tqdm(verilog_files, desc='Extracting features'):
        output_file = os.path.join(
            args.output,
            os.path.basename(vfile).replace('.v', '.feat')
        )
        
        try:
            if args.tool == 'primetime':
                extract_power_primetime(vfile, output_file)
            else:
                estimate_power(vfile, output_file)
            
            # Extract area
            extract_area_yosys(vfile, output_file)
            
        except Exception as e:
            print(f"\nError processing {vfile}: {e}")
            continue
    
    print(f"\nFeature extraction complete. Saved to {args.output}")


if __name__ == '__main__':
    main()