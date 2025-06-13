import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

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
    
# Example usage
def create_model_and_test(json_path):
    # Create model
    model = EinsteinNet(hidden_dim=64, num_classes=3)
    
    # Load real data from SM.json using the convert_to_graph function
    from .load_data import convert_to_graph
    
    # Convert JSON to HeteroData
    data = convert_to_graph(json_path)
    
    print("Graph Data:")
    print(f"Number of particles: {data['particle'].x.shape[0]}")
    print(f"Number of fields: {data['field'].x.shape[0]}")
    print(f"Number of interactions: {data['interaction'].x.shape[0]}")
    print(f"Number of particle->field edges: {data[('particle', 'in', 'field')].edge_index.shape[1]}")
    print(f"Number of field->interaction edges: {data[('field', 'involve', 'interaction')].edge_index.shape[1]}")
    
    # Adjust field features to be 10-dimensional (they're 12-dimensional in the original data)
    # This keeps compatibility with the model
    if data['field'].x.shape[1] > 10:
        data['field'].x = data['field'].x[:, :10]
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(data)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1)
    
    print(f"Output logits: {output}")
    print(f"Probabilities: {probabilities}")
    print(f"Predicted class: {predicted_class.item()}")
    
    return model

# Training function
def train_model(model, data_loader, num_epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, labels in data_loader:
            optimizer.zero_grad()
            
            output = model(batch_data)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(data_loader):.4f}')

if __name__ == "__main__":
    # Test the model
    pass