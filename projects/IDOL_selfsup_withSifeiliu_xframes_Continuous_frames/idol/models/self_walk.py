import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

EPS = 1e-20

class CRW(nn.Module):
    def __init__(self,):
        super(CRW, self).__init__()

        self.edgedrop_rate = 0.1
        self.temperature = 0.07
        self.xent = nn.CrossEntropyLoss(reduction="none")
        self._xent_targets = dict()
        self.dropout = nn.Dropout(p=self.edgedrop_rate, inplace=False)

        self.sk_targets=False

    def affinity(self, x1, x2):
        in_t_dim = x1.ndim
        if in_t_dim < 4:  # add in time dimension if not there
            x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)

        A = torch.einsum('bctn,bctm->btnm', x1, x2)

        return A.squeeze(1) if in_t_dim < 4 else A
    
    def stoch_mat(self, A, do_dropout=True):
        ''' Affinity -> Stochastic Matrix '''
        A[torch.rand_like(A) < self.edgedrop_rate] = -1e20

        return F.softmax(A/self.temperature, dim=-1)
        
    # # 特征提取
    # def pixels_to_nodes(self, x):
    #     ''' 
    #         pixel maps -> node embeddings 
    #         Handles cases where input is a list of patches of images (N>1), or list of whole images (N=1)

    #         Inputs:
    #             -- 'x' (B x N x C x T x h x w), batch of images
    #         Outputs:
    #             -- 'feats' (B x C x T x N), node embeddings
    #             -- 'maps'  (B x N x C x T x H x W), node feature maps
    #     '''
    #     B, N, C, T, h, w = x.shape

    #     #x.flatten(0, 1) 表示降纬，从0-1展平成0维，其余不变
    #     #self.encoder 表示特征提取，如res50之后的结果
    #     maps = self.encoder(x.flatten(0, 1))
        
    #     H, W = maps.shape[-2:]

    #     if N == 1:  # flatten single image's feature map to get node feature 'maps'
    #         #contiguous() 断开与原始tensor的联系，开辟新的空间位置
    #         maps = maps.permute(0, -2, -1, 1, 2).contiguous()
    #         maps = maps.view(-1, *maps.shape[3:])[..., None, None]
    #         N, H, W = maps.shape[0] // B, 1, 1

    #     # compute node embeddings by spatially pooling node feature maps
    #     feats = maps.sum(-1).sum(-1) / (H*W)
    #     feats = self.selfsim_fc(feats.transpose(-1, -2)).transpose(-1,-2)
    #     feats = F.normalize(feats, p=2, dim=1) #数据归一化
    
    #     feats = feats.view(B, N, feats.shape[1], T).permute(0, 2, 3, 1)
    #     maps  =  maps.view(B, N, *maps.shape[1:])

    #     return feats, maps

    def forward(self, q):
        '''
        Input is B x T x N*C x H x W, where either
           N>1 -> list of patches of images
           N=1 -> list of images
        '''
        # B, T, C, H, W = x.shape
        # _N, C = C//3, 3
    
        #################################################################
        # Pixels to Nodes 
        #################################################################
        # x = x.transpose(1, 2).view(B, _N, C, T, H, W)
        # q, mm = self.pixels_to_nodes(x) # ===》feats, maps
        B, C, T, N = q.shape

        #################################################################
        # Compute walks 
        #################################################################
        walks = dict()
        As = self.affinity(q[:, :, :-1], q[:, :, 1:])
        A12s = [self.stoch_mat(As[:, i], do_dropout=True) for i in range(T-1)]

        #################################################### Palindromes  回文结构
        if not self.sk_targets:  #not false
            A21s = [self.stoch_mat(As[:, i].transpose(-1, -2), do_dropout=True) for i in range(T-1)]
            AAs = []
            for i in list(range(1, len(A12s))):
                g = A12s[:i+1] + A21s[:i+1][::-1]
                aar = aal = g[0]
                for _a in g[1:]:
                    aar, aal = aar @ _a, _a @ aal

                AAs.append((f"r{i}", aar))
    
            for i, aa in AAs:
                walks[f"cyc {i}"] = [aa, self.xent_targets(aa)]

        #################################################################
        # Compute loss 
        #################################################################
        # xents = [torch.tensor([0.]).to(self.args.device)]
        xents = [torch.tensor([0.]).to(q)]
        # diags = dict()

        for name, (A, target) in walks.items():
            logits = torch.log(A+EPS).flatten(0, -2)
            loss = self.xent(logits, target).mean()
            # acc = (torch.argmax(logits, dim=-1) == target).float().mean()
            # diags.update({f"{H} xent {name}": loss.detach(),
            #               f"{H} acc {name}": acc})
            xents += [loss]

        loss = sum(xents)/max(1, len(xents)-1)
        # return q, loss, diags
        return q, loss

    def xent_targets(self, A):
        B, N = A.shape[:2]
        key = '%s:%sx%s' % (str(A.device), B,N)

        if key not in self._xent_targets:
            I = torch.arange(A.shape[-1])[None].repeat(B, 1)
            self._xent_targets[key] = I.view(-1).to(A.device)

        return self._xent_targets[key]

