import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp




class MBNet(nn.Module):
    def __init__(self, embed_dim=256, num_features_xd=78, output_dim=256, gat_dim=128, dropout=0.2):
        super(MBNet, self).__init__()
        proj_dim = 256
        self.relu = nn.ReLU()
        dropout_rate = 0.2

        # Drug structure embedding using GATConv layers
        self.gat1 = GATConv(num_features_xd, gat_dim, heads=10, dropout=dropout)
        self.gat2 = GATConv(gat_dim * 10, gat_dim, dropout=dropout)
        self.fc_gat = nn.Linear(gat_dim, output_dim)

        # Drug fingerprint embeddings (MACCS, ECFP4, RDKit)
        self.projection_mixfp = nn.Sequential(
            nn.Linear(3239, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate)
        )

        # BERT embeddings for SMILES
        self.projection_bert = nn.Sequential(
            nn.Linear(768, proj_dim),  # Corrected input dimension to match BERT per compound
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate)
        )


        # Combining structure, fingerprint, and SMILES features
        self.combine_function = nn.Linear(proj_dim * 3, proj_dim, bias=False)

        # Final output layer
        self.transform = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, data):
        # Extract data components
        x, edge_index, batch, mixfp, bert = data.x, data.edge_index, data.batch, data.mixfp, data.bert

        # Ensure mixfp and bert have the correct shape
        if len(mixfp.shape) == 1:
            mixfp = mixfp.view(-1, 3239)  # Reshape to [batch_size, 3239]
        if len(bert.shape) == 1:
            bert = bert.view(-1, 768)  # Reshape to [batch_size, 768]

        # Drug structure embedding using GAT layers
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = gmp(x, batch)  # Global max pooling
        x = self.fc_gat(x)
        structure_vector = self.relu(x)

        # Drug fingerprint embedding
        mixfp_vector = self.projection_mixfp(mixfp)

        # BERT embedding for SMILES
        bert_vector = self.projection_bert(bert)

        # Combine structure, fingerprint, and BERT features
        all_features = torch.cat([structure_vector, mixfp_vector, bert_vector], dim=-1)
        # print("All features shape:", all_features.shape)  # Debugging line to check dimensions

        # Apply the combine_function to the concatenated features
        combined_feature = self.combine_function(all_features)

        # Output prediction
        out = self.transform(combined_feature)

        return out, combined_feature
    



