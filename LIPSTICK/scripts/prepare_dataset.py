
## LIPSTICK/scripts/prepare_dataset.py

```python
"""Prepare LIPSTICK dataset from raw benchmarks."""

import os
import sys
import argparse
import subprocess
from tqdm import tqdm
import numpy as np

sys.path.append('..')
from utils.netlist_parser import convert_bench_to_verilog, lock_circuit, extract_er


def main():
    parser = argparse.ArgumentParser(description='Prepare LIPSTICK dataset')
    parser.add_argument('--benchmarks', type=str, required=True,
                       help='Directory with original .bench files')
    parser.add_argument('--output', type=str, default='../benchmarks/ISCAS85/netlists',
                       help='Output directory')
    parser.add_argument('--key-size', type=int, default=64)
    parser.add_argument('--num-wrong-keys', type=int, default=10)
    parser.add_argument('--resynthesis', type=int, default=10)
    parser.add_argument('--extract-er', action='store_true')
    args = parser.parse_args()
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.output, split), exist_ok=True)
    
    # Get benchmark files
    bench_files = [f for f in os.listdir(args.benchmarks) if f.endswith('.bench')]
    
    # Lock types
    lock_types = ['xor', 'mux', 'lut', 'sar', 'antisat', 'ble', 'unsail']
    
    total_files = 0
    
    for bench_file in tqdm(bench_files, desc='Processing benchmarks'):
        benchmark = bench_file.replace('.bench', '')
        bench_path = os.path.join(args.benchmarks, bench_file)
        
        # Convert to Verilog
        orig_verilog = convert_bench_to_verilog(bench_path)
        
        for lock_type in lock_types:
            # Lock circuit with correct key
            locked_verilog, correct_key = lock_circuit(
                orig_verilog, lock_type, args.key_size
            )
            
            # Generate wrong keys
            wrong_keys = []
            for _ in range(args.num_wrong_keys):
                wrong_key = ''.join(str(np.random.randint(0, 2)) for _ in range(args.key_size))
                # Ensure it's different from correct key
                while wrong_key == correct_key:
                    wrong_key = ''.join(str(np.random.randint(0, 2)) for _ in range(args.key_size))
                wrong_keys.append(wrong_key)
            
            keys = [correct_key] + wrong_keys
            
            # Create resynthesized versions
            for resynth in range(args.resynthesis):
                for key in keys:
                    # Extract ER if requested
                    if args.extract_er:
                        er = extract_er(orig_verilog, locked_verilog, key, correct_key)
                    else:
                        er = 0.0 if key == correct_key else np.random.random()
                    
                    # Determine split (70% train, 15% val, 15% test)
                    rand_val = np.random.random()
                    if rand_val < 0.7:
                        split = 'train'
                    elif rand_val < 0.85:
                        split = 'val'
                    else:
                        split = 'test'
                    
                    # Save netlist
                    output_file = os.path.join(
                        args.output, split,
                        f'{benchmark}_{lock_type}_{key}_{er:.4f}_r{resynth}.v'
                    )
                    
                    # Apply resynthesis
                    resynthesized = apply_resynthesis(locked_verilog, resynth)
                    
                    with open(output_file, 'w') as f:
                        f.write(resynthesized)
                    
                    total_files += 1
    
    print(f'\nDataset preparation complete!')
    print(f'Total files created: {total_files}')
    print(f'Train: {len(os.listdir(os.path.join(args.output, "train")))}')
    print(f'Val: {len(os.listdir(os.path.join(args.output, "val")))}')
    print(f'Test: {len(os.listdir(os.path.join(args.output, "test")))}')


def apply_resynthesis(verilog_code, seed):
    """Apply resynthesis using ABC with different seeds."""
    # Save to temp file
    temp_input = f'temp_input_{seed}.v'
    temp_output = f'temp_output_{seed}.v'
    
    with open(temp_input, 'w') as f:
        f.write(verilog_code)
    
    # ABC script for resynthesis
    abc_script = f"""
    read_verilog {temp_input}
    strash
    resub -K 6
    rewrite -l
    resub -K 6 -N 2
    refactor -l
    balance
    write_verilog {temp_output}
    quit
    """
    
    script_file = f'abc_script_{seed}.txt'
    with open(script_file, 'w') as f:
        f.write(abc_script)
    
    try:
        subprocess.run(['abc', '-f', script_file], check=True, capture_output=True)
        
        with open(temp_output, 'r') as f:
            resynthesized = f.read()
        
        # Cleanup
        os.remove(temp_input)
        os.remove(temp_output)
        os.remove(script_file)
        
        return resynthesized
    except:
        # If ABC fails, return original
        os.remove(temp_input)
        if os.path.exists(script_file):
            os.remove(script_file)
        return verilog_code


if __name__ == '__main__':
    main()
```
