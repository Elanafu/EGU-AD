# ========== paste into model/temporal_encoder.py ==========
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TemporalConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size,
                              dilation=dilation, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.PReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                 # x: [T, Din]
        x3 = x.unsqueeze(0).permute(0, 2, 1)      # [1, Din, T]
        y = self.conv(x3)                          # [1, Dout, T]
        y = self.bn(y); y = self.act(y); y = self.drop(y)
        y = y.permute(0, 2, 1).squeeze(0)         # [T, Dout]
        return y

class _GateRouter(nn.Module):
    """基于局部形状统计的门控，产生每时刻对各分支的 logits（再做 softmax）"""
    def __init__(self, in_feats=6, hidden=32, branches=3, win_large=31):
        super().__init__()
        self.win_large = win_large
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, branches)  # 输出对每个分支的logits
        )

    @staticmethod
    def _rollwin(x, k):
        pad = (k - 1) // 2
        return F.pad(x, (pad, pad), mode='replicate').unfold(-1, k, 1)

    def forward(self, s_bt):  # s_bt: [1,T] —— 用于路由的标量轨迹（融合前的基准强度）
        B, T = s_bt.shape
        d1 = F.pad(torch.abs(s_bt[:, 1:] - s_bt[:, :-1]), (1, 0))
        d2 = F.pad(torch.abs(s_bt[:, 2:] - 2*s_bt[:, 1:-1] + s_bt[:, :-2]), (1, 1))
        seg = self._rollwin(s_bt, self.win_large)
        mu = seg.mean(-1)
        var = seg.var(-1, unbiased=False) + 1e-6
        kurt = ((seg - mu.unsqueeze(-1)).pow(4).mean(-1) / (var**2)).clamp(max=20.0)
        feat = torch.stack([s_bt, d1, d2, mu, var, kurt], dim=-1)  # [1,T,6]
        logits = self.mlp(feat).squeeze(0)                         # [T,B]
        return logits                                              # 每时刻对各分支的logits

class AdaptiveMSEarlyFuse(nn.Module):
    """
    自适应多尺度（特征层早融合）：
    - 多个 (k,d) 分支 -> 通道归一化 -> 门控权重（softmax(temperature)） -> 加权求和
    """
    def __init__(self, in_dim, out_dim,
                 branches=((3,1),(5,2),(7,3)),
                 dropout=0.2,
                 router_win=31,
                 kappa_min=0.15, kappa_max=1.0):
        super().__init__()
        self.branches = branches
        self.num_branches = len(branches)
        self.blocks = nn.ModuleList([
            TemporalConvBlock(in_dim, out_dim, k, d, dropout)
            for (k,d) in branches
        ])
        # 对齐：把每个分支的通道统计对齐到统一分布（减少尺度漂移）
        self.align = nn.ModuleList([nn.LayerNorm(out_dim) for _ in branches])
        self.base_idx = 0  # 基准分支（建议 (3,1) 放第一位）
        self.router = _GateRouter(in_feats=6, hidden=32, branches=self.num_branches, win_large=router_win)
        self.kappa_min, self.kappa_max = kappa_min, kappa_max
        self.drop = nn.Dropout(dropout)

        self.last_alpha = None  # [T,B]

    def forward(self, x):          # x: [T, Din]
        feats = []
        strengths = []
        for blk, ln in zip(self.blocks, self.align):
            y = blk(x)                              # [T, H]
            y = ln(y)                               # 对齐分支分布
            feats.append(y)
            strengths.append(torch.sqrt((y*y).mean(dim=-1, keepdim=True)+1e-6).transpose(0,1))  # [1,T]

        # 基准强度（简单均值）作为路由的“形状信号”
        s_base = torch.stack(strengths, dim=-1).mean(dim=-1).squeeze(0)  # [T]
        s_base = s_base.unsqueeze(0)  # [1,T]

        # 路由 logits -> 温度化 softmax
        logits = self.router(s_base)                  # [T,B]
        # 温度 κ(·)：α大（尖峰）→ 小温度（更接近argmax）；宽帽→大温度（更均匀）
        # 这里用 logits 的幅度代理 α：|∇s| + |∇²s|
        with torch.no_grad():
            d1 = F.pad(torch.abs(s_base[:,1:]-s_base[:,:-1]), (1,0))
            d2 = F.pad(torch.abs(s_base[:,2:]-2*s_base[:,1:-1]+s_base[:,:-2]), (1,1))
            alpha = (0.5*d1 + 0.5*d2).clamp(0, 10.0) / 10.0   # [1,T] -> [0,1] 近似
        kappa = self.kappa_min + (1.0 - alpha) * (self.kappa_max - self.kappa_min)  # [1,T]
        logits_t = logits / (kappa.squeeze(0).unsqueeze(-1) + 1e-6)                 # [T,B]
        w = torch.softmax(logits_t, dim=-1)                                          # [T,B]
        self.last_alpha = w.detach()

        # 残差保护：保留基准分支的峰值
        y_base = feats[self.base_idx]                                                # [T,H]
        y_fuse = 0.0
        for b in range(self.num_branches):
            wb = w[:, b].unsqueeze(-1)                                               # [T,1]
            y_fuse = y_fuse + feats[b] * wb
        # 残差注入（小系数，避免过平滑）
        y = y_fuse + 0.2 * y_base
        y = self.drop(y)
        return y  # [T,H]


class TemporalGraphEncoder(nn.Module):
    """
    早融合多尺度 -> GAT -> 投影
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        # 1) 早融合输出维度固定为 hidden_dim
        self.temporal_ms = AdaptiveMSEarlyFuse(
            in_dim=input_dim, out_dim=hidden_dim,
            branches=((3,1),(5,2),(7,3)),
            # dropout=dropout, router_win=31, kappa_min=0.15, kappa_max=1.0
            dropout=dropout, router_win=31, kappa_min=0.15, kappa_max=1.0
        )

        # 2) GAT：非 lazy；确保 out_channels * heads == hidden_dim
        heads = 4
        assert hidden_dim % heads == 0, \
            f"hidden_dim={hidden_dim} 必须能被 heads={heads} 整除"
        self.gat_layers = nn.ModuleList([
            GATConv(in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True)        # concat=True -> 输出维度 = out_channels*heads = hidden_dim
            for _ in range(num_layers)
        ])

        # 残差和投影
        self.skip_proj = nn.Linear(input_dim, hidden_dim)
        self.pre_gat_ln = nn.LayerNorm(hidden_dim)

        # 标量门控（稳定融合 h0 与 GAT 消息）
        self.gat_res_gate = nn.Parameter(torch.zeros(1))

        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x, edge_index):
        # 残差支路
        skip_val = self.skip_proj(x)            # [T, H]

        # 早融合（保证 [T, H]）
        h0 = self.temporal_ms(x)                # [T, H]
        h0 = self.pre_gat_ln(h0)

        # GAT 堆叠（每层输出 [T, H]）
        g = h0
        for gat in self.gat_layers:
            g_msg = gat(g, edge_index)          # [T, H]
            g = F.elu(g_msg)

        # 标量门控融合（避免通道级广播问题）
        mix = torch.sigmoid(self.gat_res_gate)  # 标量 (0,1)
        g = (1.0 - mix) * h0 + mix * g          # [T, H]

        # 残差
        h_out = g + skip_val                    # [T, H]

        # 时间状态 p
        p_out = self.proj_head(g)               # [T, H]
        return h_out, p_out

