"""Power/area extraction, in case you don't want to use simulation tools.

"""
## GALA/utils/power_area_extractor.py

"""Power and area feature extraction utilities."""

import os
import subprocess
import numpy as np
import re


def extract_power_primetime(verilog_file, output_file):
    """Extract power consumption using Synopsys PrimeTime PX.
    
    Args:
        verilog_file: Path to Verilog netlist
        output_file: Path to save power features
    
    Returns:
        Dictionary with power metrics
    """
    # Create PrimeTime script
    pt_script = f"""
    read_verilog {verilog_file}
    link_design
    read_sdc constraints.sdc
    update_timing
    report_power -hierarchy -verbose > power_report.txt
    quit
    """
    
    script_file = 'run_primetime.tcl'
    with open(script_file, 'w') as f:
        f.write(pt_script)
    
    try:
        # Run PrimeTime
        subprocess.run(['pt_shell', '-f', script_file], check=True, capture_output=True)
        
        # Parse power report
        power_data = parse_power_report('power_report.txt')
        
        # Save features
        with open(output_file, 'w') as f:
            f.write(f"total_power: {power_data['total_power']}\n")
            f.write(f"static_power: {power_data['static_power']}\n")
            f.write(f"dynamic_power: {power_data['dynamic_power']}\n")
            
            # Per-gate power
            for gate, power in power_data['gate_power'].items():
                f.write(f"gate_{gate}: {power}\n")
        
        return power_data
    
    except Exception as e:
        print(f"Error running PrimeTime: {e}")
        print("Falling back to estimation...")
        return estimate_power(verilog_file, output_file)


def parse_power_report(report_file):
    """Parse PrimeTime power report."""
    power_data = {
        'total_power': 0.0,
        'static_power': 0.0,
        'dynamic_power': 0.0,
        'gate_power': {}
    }
    
    with open(report_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Extract total power
        if 'Total Power' in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                power_data['total_power'] = float(match.group(1))
        
        # Extract static power
        if 'Leakage Power' in line or 'Static Power' in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                power_data['static_power'] = float(match.group(1))
        
        # Extract dynamic power
        if 'Dynamic Power' in line or 'Switching Power' in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                power_data['dynamic_power'] = float(match.group(1))
        
        # Extract per-gate power
        if any(gate in line for gate in ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'NOT', 'BUF', 'MUX']):
            parts = line.split()
            if len(parts) >= 2:
                gate_name = parts[0]
                try:
                    gate_power = float(parts[-1])
                    power_data['gate_power'][gate_name] = gate_power
                except:
                    pass
    
    return power_data


def estimate_power(verilog_file, output_file):
    """Estimate power consumption based on gate types and counts.
    
    This is a fallback when PrimeTime is not available.
    """
    # Gate power estimates (normalized)
    gate_power = {
        'and': 1.0,
        'or': 1.0,
        'xor': 1.5,
        'nand': 0.9,
        'nor': 0.9,
        'xnor': 1.4,
        'not': 0.5,
        'buf': 0.4,
        'mux': 2.0
    }
    
    # Count gates
    with open(verilog_file, 'r') as f:
        content = f.read().lower()
    
    gate_counts = {}
    for gate_type in gate_power.keys():
        count = content.count(gate_type)
        gate_counts[gate_type] = count
    
    # Estimate total power
    total_dynamic = sum(gate_power[gt] * cnt for gt, cnt in gate_counts.items())
    total_static = total_dynamic * 0.3  # Assume 30% leakage
    total_power = total_dynamic + total_static
    
    # Normalize
    total_power = total_power / 1000.0  # Scale down
    
    # Save features
    with open(output_file, 'w') as f:
        f.write(f"total_power: {total_power}\n")
        f.write(f"static_power: {total_static / 1000.0}\n")
        f.write(f"dynamic_power: {total_dynamic / 1000.0}\n")
        
        # Per-gate estimates
        for gate_type, count in gate_counts.items():
            avg_power = gate_power[gate_type] * count / max(sum(gate_counts.values()), 1)
            f.write(f"gate_{gate_type}: {avg_power}\n")
    
    return {
        'total_power': total_power,
        'static_power': total_static / 1000.0,
        'dynamic_power': total_dynamic / 1000.0,
        'gate_power': {gt: gate_power[gt] * gate_counts[gt] for gt in gate_power}
    }


def extract_area_yosys(verilog_file, output_file):
    """Extract area using Yosys synthesis.
    
    Args:
        verilog_file: Path to Verilog netlist
        output_file: Path to save area features
    
    Returns:
        Area value
    """
    yosys_script = f"""
    read_verilog {verilog_file}
    hierarchy -check
    proc; opt; memory; opt
    techmap; opt
    stat
    """
    
    script_file = 'run_yosys.ys'
    with open(script_file, 'w') as f:
        f.write(yosys_script)
    
    try:
        result = subprocess.run(['yosys', '-s', script_file], 
                              capture_output=True, text=True, check=True)
        
        # Parse area from stat output
        area = parse_yosys_area(result.stdout)
        
        with open(output_file, 'a') as f:
            f.write(f"total_area: {area}\n")
        
        return area
    
    except Exception as e:
        print(f"Error running Yosys: {e}")
        return estimate_area(verilog_file, output_file)


def parse_yosys_area(yosys_output):
    """Parse area from Yosys output."""
    # Look for chip area in stat output
    for line in yosys_output.split('\n'):
        if 'Chip area' in line:
            match = re.search(r'(\d+\.?\d*)', line)
            if match:
                return float(match.group(1))
    
    # Estimate from gate counts
    gate_count = 0
    for line in yosys_output.split('\n'):
        if 'Number of cells' in line:
            match = re.search(r'(\d+)', line)
            if match:
                gate_count = int(match.group(1))
    
    return gate_count * 10.0  # Rough estimate


def estimate_area(verilog_file, output_file):
    """Estimate area based on gate counts."""
    # Gate area estimates
    gate_area = {
        'and': 3.0,
        'or': 3.0,
        'xor': 6.0,
        'nand': 2.5,
        'nor': 2.5,
        'xnor': 5.5,
        'not': 1.0,
        'buf': 2.0,
        'mux': 8.0
    }
    
    with open(verilog_file, 'r') as f:
        content = f.read().lower()
    
    total_area = 0
    for gate_type, area in gate_area.items():
        count = content.count(gate_type)
        total_area += area * count
    
    with open(output_file, 'a') as f:
        f.write(f"total_area: {total_area}\n")
    
    return total_area
