
## LIPSTICK/utils/graph_utils.py

"""Graph construction utilities."""

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def netlist_to_graph(verilog_file):
    """Convert Verilog netlist to PyG graph.
    
    Args:
        verilog_file: Path to Verilog file
    
    Returns:
        PyG Data object
    """
    # Parse Verilog and build NetworkX graph
    G = parse_verilog_to_networkx(verilog_file)
    
    # Add node features
    node_features = []
    for node in G.nodes():
        feat = get_node_features(G, node)
        node_features.append(feat)
    
    # Convert to PyG format
    data = from_networkx(G)
    data.x = torch.tensor(node_features, dtype=torch.float32)
    
    return data


def parse_verilog_to_networkx(verilog_file):
    """Parse Verilog netlist to NetworkX graph.
    
    Args:
        verilog_file: Path to Verilog file
    
    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()
    
    with open(verilog_file, 'r') as f:
        lines = f.readlines()
    
    node_id = 0
    wire_to_node = {}
    
    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines
        if line.startswith('//') or not line:
            continue
        
        # Parse module declaration
        if line.startswith('module'):
            continue
        
        # Parse input/output declarations
        if line.startswith('input') or line.startswith('output'):
            wires = line.split()[1:]
            for wire in wires:
                wire = wire.rstrip(';,')
                if wire not in wire_to_node:
                    G.add_node(node_id, wire=wire, gate_type='port', 
                              is_input=line.startswith('input'),
                              is_output=line.startswith('output'))
                    wire_to_node[wire] = node_id
                    node_id += 1
        
        # Parse gate instantiations
        if any(gate in line for gate in ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'NOT', 'BUF', 'MUX']):
            parts = line.replace('(', ' ').replace(')', ' ').replace(',', ' ').replace(';', '').split()
            gate_type = parts[0].lower()
            output_wire = parts[1]
            input_wires = parts[2:]
            
            # Create gate node
            if output_wire not in wire_to_node:
                G.add_node(node_id, wire=output_wire, gate_type=gate_type, 
                          is_input=False, is_output=False)
                wire_to_node[output_wire] = node_id
                gate_node = node_id
                node_id += 1
            else:
                gate_node = wire_to_node[output_wire]
                G.nodes[gate_node]['gate_type'] = gate_type
            
            # Create edges from inputs to gate
            for input_wire in input_wires:
                if input_wire not in wire_to_node:
                    G.add_node(node_id, wire=input_wire, gate_type='wire',
                              is_input=False, is_output=False)
                    wire_to_node[input_wire] = node_id
                    input_node = node_id
                    node_id += 1
                else:
                    input_node = wire_to_node[input_wire]
                
                G.add_edge(input_node, gate_node)
    
    return G


def get_node_features(G, node):
    """Extract feature vector for a node.
    
    Features:
    - One-hot encoding of gate type (10 types)
    - Is primary input (1)
    - Is primary output (1)
    - In-degree (1)
    - Out-degree (1)
    - Total: 14 features
    
    Args:
        G: NetworkX graph
        node: Node ID
    
    Returns:
        Feature vector as list
    """
    gate_types = ['and', 'or', 'xor', 'nand', 'nor', 'xnor', 'not', 'buf', 'mux', 'port', 'wire']
    gate_type = G.nodes[node].get('gate_type', 'wire')
    
    # One-hot encode gate type
    gate_feat = [1 if gate_type == gt else 0 for gt in gate_types]
    
    # Additional features
    is_input = 1 if G.nodes[node].get('is_input', False) else 0
    is_output = 1 if G.nodes[node].get('is_output', False) else 0
    in_degree = G.in_degree(node)
    out_degree = G.out_degree(node)
    
    return gate_feat + [is_input, is_output, in_degree, out_degree]
