import json
import numpy as np
import torch
import torch_geometric.data as Data
from typing import Dict, List, Tuple, Any, Union
from torch_geometric.data import HeteroData

import os
import sys

def convert_to_graph(json_path: str) -> Data.HeteroData:
    ptcl_types = {"real": 1, "complex": 2, "psuedo": 3, "fermion": 4, "vector": 5}
    interaction_types = {"DC": 1, "yukawa": 2}
    
    def convert_rep(rep: Union[str, int]) -> int:
        if isinstance(rep, str):
            rep_map = {"singlet": 1, "fnd": 2, "adj": 3}
            return rep_map[rep]
        else:
            return rep
    
    def convert_chirality(chirality: Union[str, None]) -> int:
        if chirality == "left": return 1
        if chirality == "right": return -1
        if chirality is None: return 0
        
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create a heterogeneous graph
    hetero_data = HeteroData()
    
    # Track mappings from original IDs to node indices for each type
    particle_id_map = {}
    field_id_map = {}
    interaction_id_map = {}
    
    # --------- particles ---------
    particle_features = []
    for idx, p in enumerate(data["particles"]):
        particle_id_map[p["id"]] = idx
        p_feature = [
            ptcl_types.get(p["type"], 0),
            p["mass"],
            p["charge"]
        ]
        particle_features.append(p_feature)
    
    if particle_features:
        hetero_data["particle"].x = torch.tensor(particle_features, dtype=torch.float)
    else:
        hetero_data["particle"].x = torch.zeros((0, 3), dtype=torch.float)
    
    # --------- fields ---------
    field_features = []
    for idx, f in enumerate(data["fields"]):
        field_id_map[f["id"]] = idx
        
        # Base features
        f_feature = [
            ptcl_types.get(f["type"], 0),
            f["dim"],
            f["gen"],
            convert_chirality(f["chirality"]),
            int(f["self_conjugate"]),
        ]
        f_feature.extend(f["QuantumNumber"].values())
        f_feature.extend([convert_rep(rep) for rep in f["reps"].values()])
        field_features.append(f_feature)
    if field_features:
        hetero_data["field"].x = torch.tensor(field_features, dtype=torch.float)
    else:
        hetero_data["field"].x = torch.zeros((0, 12), dtype=torch.float)
    
    # --------- interactions ---------
    interaction_features = []
    for idx, i in enumerate(data["interactions"]):
        interaction_id_map[i["id"]] = idx
        i_feature = [interaction_types.get(i["type"], 0)]
        interaction_features.append(i_feature)
    
    if interaction_features:
        hetero_data["interaction"].x = torch.tensor(interaction_features, dtype=torch.float)
    else:
        hetero_data["interaction"].x = torch.zeros((0, 1), dtype=torch.float)
    
    # --------- edges ---------
    # Particle to Field edges
    particle_field_edges = []
    field_particle_edges = []
    
    for field_idx, field in enumerate(data["fields"]):
        if "particles" in field:
            for particle_id in field["particles"]:
                if particle_id in particle_id_map:
                    particle_idx = particle_id_map[particle_id]
                    particle_field_edges.append([particle_idx, field_idx])
                    field_particle_edges.append([field_idx, particle_idx])
    if particle_field_edges:
        hetero_data["particle", "in", "field"].edge_index = torch.tensor(
            particle_field_edges, dtype=torch.long
        ).t().contiguous()
    if field_particle_edges:
        hetero_data["field", "rev_in", "particle"].edge_index = torch.tensor(
            field_particle_edges, dtype=torch.long
        ).t().contiguous()
    
    # Field to Interaction edges
    field_interaction_edges = []
    interaction_field_edges = []
    
    for interaction_idx, interaction in enumerate(data["interactions"]):
        if "fields" in interaction:
            for field_id in interaction["fields"]:
                if field_id in field_id_map:
                    field_idx = field_id_map[field_id]
                    field_interaction_edges.append([field_idx, interaction_idx])
                    interaction_field_edges.append([interaction_idx, field_idx])

    if field_interaction_edges:
        hetero_data["field", "involve", "interaction"].edge_index = torch.tensor(
            field_interaction_edges, dtype=torch.long
        ).t().contiguous() 
    if interaction_field_edges:
        hetero_data["interaction", "rev_involve", "field"].edge_index = torch.tensor(
            interaction_field_edges, dtype=torch.long
        ).t().contiguous()

    return hetero_data

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    json_path = "SM.json"
    graph = convert_to_graph(json_path)
    print(graph)
    