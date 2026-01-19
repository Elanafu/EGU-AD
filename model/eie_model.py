import torch
import torch.nn as nn
import torch.nn.functional as F

from model.temporal_encoder import (
    TemporalGraphEncoder, 
)

# -------- Energy Head 保持不变 --------
class EnergyHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, h):  # h: [T,H]
        return self.mlp(h) # [T,1]


class EIEModel(nn.Module):
    """
    EIE-AD 主干（采用“早融合自适应多尺度”的 TemporalGraphEncoder）。
    forward(data) 返回:
      h:        (T, H)
      p:        (T, H)
      E:        (T,)
      dE_pred:  (T,)
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int):
        super().__init__()

        # 早融合多尺度 在 TemporalGraphEncoder 内部完成
        self.encoder = TemporalGraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # 能量头 h -> E
        self.energy_head = EnergyHead(hidden_dim)

        # 局部能量流动头：用 (h_t, h_{t-1}) 预测 dE
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )

        proj_dim = 128
        self.proj_T = nn.Sequential(          # 时间/预测分支投影 (用 p 或 h)
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        self.proj_G = nn.Sequential(          # 能量-几何分支投影 (用 [kappa, r, dE] 特征)
            nn.Linear(3, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        self.pgw = PhysicsGuidedWeights(
            wR_prior=0.50,
            wK_prior=0.25,
            wMS_prior=0.25,
            wRank_prior=0.70,
        )

    def _predict_dE(self, h: torch.Tensor) -> torch.Tensor:
        """
        给定 h: [T,H]，用 flow_head 预测 dE: [T,]
        """
        T = h.size(0)
        if T > 1:
            h_t   = h[1:]                              # (T-1,H)
            h_tm1 = h[:-1]                             # (T-1,H)
            pair  = torch.cat([h_t, h_tm1], dim=-1)    # (T-1, 2H)
            dE_pred_inner = self.flow_head(pair).squeeze(-1)  # (T-1,)
            dE_pred = F.pad(dE_pred_inner, (1, 0))            # (T,)
        else:
            dE_pred = torch.zeros(T, device=h.device, dtype=h.dtype)
        return dE_pred

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        # === 早融合多尺度编码 ===
        h, p = self.encoder(x, edge_index)             # (T,H), (T,H)

        # === 能量与能量流 ===
        E = self.energy_head(h).squeeze(-1)            # (T,)
        dE_pred = self._predict_dE(h)                  # (T,)

        return h, p, E, dE_pred


class PhysicsGuidedWeights(nn.Module):
    def __init__(self,
                 wR_prior: float = 0.50,
                 wK_prior: float = 0.25,
                 wMS_prior: float = 0.25,
                 wRank_prior: float = 0.80):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(3))
        self.theta_rank = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("w_prior", torch.tensor([wR_prior, wK_prior, wMS_prior], dtype=torch.float32))
        self.register_buffer("wRank_prior", torch.tensor(float(wRank_prior), dtype=torch.float32))

    def get_weights(self):
        w = torch.softmax(self.theta, dim=0)           # [3]
        wR, wK, wMS = w[0], w[1], w[2]
        wRank = torch.sigmoid(self.theta_rank)
        return wR, wK, wMS, wRank

    def prior_loss(self, lambda_w: float = 1e-3) -> torch.Tensor:
        wR, wK, wMS, wRank = self.get_weights()
        loss = lambda_w * (
            (wR - self.w_prior[0])**2 +
            (wK - self.w_prior[1])**2 +
            (wMS - self.w_prior[2])**2 +
            (wRank - self.wRank_prior)**2
        )
        return loss
