import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

class EinsteinNet(nn.Module):
    def __init__(self, hidden_dim = 64, num_classes = 3):
        super(EinsteinNet, self).__init__()
        
        # Node-level transformations to ensure consistent dimensions
        self.node_transforms = nn.ModuleDict({
            'particle': nn.Linear(3, hidden_dim),
            'field': nn.Linear(10, hidden_dim), 
            'interaction': nn.Linear(1, hidden_dim)
        })
        
        # After node transform, all features have hidden_dim dimensions
        self.conv1 = HeteroConv({
            ('particle', 'in', 'field'): SAGEConv(hidden_dim, hidden_dim),
            ('field', 'involve', 'interaction'): SAGEConv(hidden_dim, hidden_dim),
            ('field', 'rev_in', 'particle'): SAGEConv(hidden_dim, hidden_dim),
            ('interaction', 'rev_involve', 'field'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('particle', 'in', 'field'): SAGEConv(hidden_dim, hidden_dim),
            ('field', 'involve', 'interaction'): SAGEConv(hidden_dim, hidden_dim),
            ('field', 'rev_in', 'particle'): SAGEConv(hidden_dim, hidden_dim),
            ('interaction', 'rev_involve', 'field'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='sum')
        
        # Graph-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3 node types * hidden_dim
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        # Transform all node features to same dimension
        transformed_x = {}
        for node_type in x_dict:
            transformed_x[node_type] = self.node_transforms[node_type](x_dict[node_type])
        
        # Apply heterogeneous convolutions
        x_dict = self.conv1(transformed_x, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Global pooling for each node type (graph-level representation)
        graph_repr = []
        for node_type in ['particle', 'field', 'interaction']:
            if node_type in x_dict:
                # Simple mean pooling across nodes of each type
                pooled = torch.mean(x_dict[node_type], dim=0, keepdim=True)
                graph_repr.append(pooled)
        
        # Concatenate representations from all node types
        graph_repr = torch.cat(graph_repr, dim=1)  # Shape: [1, hidden_dim * 3]
        
        out = self.classifier(graph_repr)
        return out

if __name__ == "__main__":
    model = EinsteinNet()
    print(model)