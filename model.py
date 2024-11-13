import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes=[64,64,64,1], arl=False, norm = True, dropout=0.0):
        super().__init__()
        self.norm = norm
        if norm:
            self.batch_norm = nn.BatchNorm1d(layer_sizes[0])

        self.arl = arl
        if arl:
            self.attention = nn.Sequential(
                nn.Linear(layer_sizes[0],layer_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_sizes[0],layer_sizes[0])
            )

        self.layer_sizes = layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError()
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        if self.norm:
            x = self.batch_norm(x)
        if self.arl:
            x = x * self.attention(x)
        for layer in self.layers[:-1]:
            x = self.dropout(self.act(layer(x)))
        x = self.layers[-1](x)
        return x

class MyNet(nn.Module):
    def __init__(self, layer_sizes=[128,64,32,16,1], arl=False, norm = True, dropout=0.0):
        super().__init__()

        self.emb_manufacturer = nn.Embedding(9,16)
        self.emb_model = nn.Embedding(170,16)
        self.emb_gearbox_type = nn.Embedding(3,16)
        self.emb_fuel_type = nn.Embedding(4,16)

        self.emb_continuous = MLP([5,32,64], norm = norm, arl = arl, dropout = dropout)

        self.predictor = MLP(layer_sizes, norm = False, arl = False, dropout = dropout)

    def forward(self,x1,x2):
        emb_manufacturer = self.emb_manufacturer(x1[:,0])
        emb_model = self.emb_model(x1[:,1])
        emb_gearbox_type = self.emb_gearbox_type(x1[:,2])
        emb_fuel_type = self.emb_fuel_type(x1[:,3])

        emb_continuous = self.emb_continuous(x2)

        x = torch.concatenate([emb_manufacturer,emb_model,emb_gearbox_type,emb_fuel_type,emb_continuous],dim=-1)
        x = self.predictor(x)
        return x