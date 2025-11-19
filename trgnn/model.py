import torch
import torch.nn as nn
from .modules import TimeEncoder,MemoryUpdater,TimeProjection,GraphAttention,GraphSum

class TGAT(nn.Module):
    def __init__(self,node_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.attention=GraphAttention(node_dim=node_dim,latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim
    
    def forward(self,data_loader,device):
        """
        Input:
            data_loader: List of batch
                batch
                    raw: [B,N,1]
                    t: [B,N,1]
                    tar: [B,1]
                    n_mask: [B,N,]
            device: GPU
        Output:
            logit: [B_seq_len,B,1]
        """

        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            raw=batch['raw'] # [B,N,1], float
            t=batch['t'] # [B,N,1], float
            tar=batch['tar'] # [B,1], long
            n_mask=batch['n_mask'] # [B,N,], neighbor node mask