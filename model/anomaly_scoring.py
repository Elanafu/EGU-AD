import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


@torch.no_grad()
def compute_anomaly_scores(
    h: torch.Tensor,                       # [T, Dh] 中间表示
    p: Optional[torch.Tensor] = None,      # [T, Dp] 可能为 None
    E: Optional[torch.Tensor] = None,      # [T]    能量轨迹
    dE_pred: Optional[torch.Tensor] = None,# [T]    预测能量增量
    *,
    use_energy: bool = True,
    # Temporal 权重
    w_temporal_dev: float = 0.5,
    w_temporal_mag: float = 0.5,
    # Energy Naturalness 内部初始权重
    wR: float = 0.50,   # Trend deviation
    wK: float = 0.25,   # Residual change rate
    wMS: float = 0.25,  # Trajectory naturalness (multi-scale)
    # T-vs-E 融合
    w_rank_EG: float = 0.5,
    # 其它参数
    roll_win: int = 61,
    ms_scales: Tuple[int, ...] = (1, 4, 16),
    smooth_k_mean: int = 5,
    ema_alpha: float = 0.2,
    weights: Optional[dict] = None,
    **kwargs
) -> Tuple[torch.Tensor, Dict]:
    eps = 1e-6
    device = h.device
    T = h.shape[0]

    # -----------------------------------------------------------
    # 工具函数
    # -----------------------------------------------------------
    def winsorize(x: torch.Tensor, lo_q=1.0, hi_q=99.0) -> torch.Tensor:
        xx = x.detach().cpu().float().numpy().reshape(-1)
        lo = float(np.percentile(xx, lo_q))
        hi = float(np.percentile(xx, hi_q))
        return x.clamp(min=lo, max=hi)

    def rolling_z(x: torch.Tensor, k: int) -> torch.Tensor:
        """滚动均值方差标准化（确定性）"""
        if x.numel() <= 1:
            return (x - x.mean()) / (x.std(unbiased=False) + eps)
        k = max(3, int(k))
        if k % 2 == 0:
            k -= 1
        pad = k // 2
        x1 = F.pad(x.view(1,1,-1), (pad,pad), mode="replicate")
        w = torch.ones(1,1,k, device=x.device, dtype=x.dtype) / k
        mu  = F.conv1d(x1, w).view(-1)
        mu2 = F.conv1d(x1 * x1, w).view(-1)
        var = (mu2 - mu*mu).clamp_min(1e-6)
        return (x - mu) / (var.sqrt() + eps)

    def _ranks(x: torch.Tensor) -> torch.Tensor:
        idx = torch.argsort(x)
        r = torch.empty_like(idx, dtype=torch.float)
        r[idx] = torch.arange(x.numel(), device=x.device, dtype=torch.float)
        return r

    # ===========================================================
    # (A) Temporal Consistency Scoring
    # ===========================================================
    if p is not None and T > 1:
        # ---- 1. 状态演化强度：dp_t = ||p_t - p_{t-1}|| ----
        dp = torch.norm(p[1:] - p[:-1], p=2, dim=1)  # [T-1]
        dp = torch.cat([dp[:1], dp], dim=0)          # [T]

        # ---- 2. EMA 基线估计 ----
        ema = torch.zeros_like(dp)
        ema[0] = dp[0]
        for t in range(1, T):
            ema[t] = ema_alpha * dp[t] + (1 - ema_alpha) * ema[t-1]

        # ---- 3. 残差：状态演化偏差 ----
        time_dev_raw = (dp - ema).abs()

        # ---- 4. robust z-score ----
        median = torch.median(time_dev_raw)
        mad = torch.median(torch.abs(time_dev_raw - median)) + 1e-6
        tdev_z = (time_dev_raw - median) / mad

    else:
        # p 不存在 → 时间分支全零
        tdev_z = torch.zeros(T, device=device)
    


    # ---- 5. 幅值项 ----
    mag = torch.norm(h, p=2, dim=1)

    # ---- 6. 对幅值做 rolling z-score（幅值不是稳态序列，用 rolling 更合理）----
    mag_z = rolling_z(mag, roll_win)

    # ---- 7. 最终时间一致性得分 ----
    temporal_score = (
        w_temporal_dev * tdev_z +
        w_temporal_mag * mag_z
    )


    # ===========================================================
    # (B) Energy Naturalness Scoring
    # ===========================================================
    if use_energy and (E is not None):
        E = E.view(-1)

        # 多尺度 κ_ms(E)：轨迹自然性
        def multi_scale_kappa_ms(E1: torch.Tensor):
            outs = []
            target_len = E1.numel()  # 原始长度
            
            for s in ms_scales:
                k_mean = max(1, int(s * smooth_k_mean))
                # 稳健平滑
                Es = winsorize(E1, 1, 99)
                pad = k_mean // 2
                x1 = F.pad(Es.view(1,1,-1), (pad,pad), mode="reflect")
                w = torch.ones(1,1,k_mean, device=E1.device, dtype=E1.dtype) / k_mean
                Es = F.conv1d(x1, w).view(-1)
                
                # 确保输出长度与原始长度一致
                if Es.numel() > target_len:
                    Es = Es[:target_len]
                elif Es.numel() < target_len:
                    # 补齐到目标长度
                    Es = F.pad(Es, (0, target_len - Es.numel()), mode='replicate')

                # 一阶差分
                d1 = torch.zeros_like(Es)
                if Es.numel()>1:
                    d1[1:-1] = 0.5*(Es[2:] - Es[:-2])
                    d1[0]    = Es[1] - Es[0]
                    d1[-1]   = Es[-1]- Es[-2]

                # 二阶差分强度
                d2 = torch.zeros_like(d1)
                if d1.numel()>1:
                    d2[1:-1] = (d1[2:] - d1[:-2]).abs()
                    d2[0]    = (d1[1] - d1[0]).abs()
                    d2[-1]   = (d1[-1] - d1[-2]).abs()

                d2 = winsorize(d2, 1, 99)
                q95 = float(np.percentile(d2.cpu().numpy(), 95)) + eps
                outs.append(torch.tanh(d2/q95))

            return torch.stack(outs, dim=0).max(dim=0).values

        kappa_ms = multi_scale_kappa_ms(E)
        kms_z    = rolling_z(kappa_ms, roll_win)

        # 若 dE_pred 存在，能量趋势残差
        if dE_pred is not None:
            dE_pred = dE_pred.view(-1)

            if T > 1:
                dE_true = torch.zeros_like(E)
                dE_true[1:] = E[1:] - E[:-1]
                dE_true[0]  = dE_true[1]
                r_raw = (dE_true - dE_pred).abs()
            else:
                r_raw = torch.zeros_like(E)

            r_z = rolling_z(winsorize(r_raw, 1, 99), roll_win)

            # kappa_z：残差变化率
            if r_raw.numel()>1:
                k_raw = torch.zeros_like(r_raw)
                k_raw[1:] = (r_raw[1:] - r_raw[:-1]).abs()
            else:
                k_raw = torch.zeros_like(r_raw)

            kappa_z = rolling_z(winsorize(k_raw, 1,99), roll_win)

            # E 分支总分
            eg_score = wR * r_z + wK * kappa_z + wMS * kms_z
        else:
            r_z = torch.zeros_like(kms_z)
            kappa_z = torch.zeros_like(kms_z)
            eg_score = kms_z
    else:
        eg_score = torch.zeros(T, device=device)
        r_z = kappa_z = kms_z = torch.zeros(T, device=device)

    # ===========================================================
    # (C) T vs E 的秩融合 (Late Rank Fusion)
    # ===========================================================
    w_rank_T = 1 - w_rank_EG
    rank_T = _ranks(temporal_score)
    rank_E = _ranks(eg_score)

    rank_avg = (w_rank_T * rank_T + w_rank_EG * rank_E) / (w_rank_T + w_rank_EG + eps)
    final = (rank_avg - rank_avg.mean()) / (rank_avg.std(unbiased=False) + eps)

    # -----------------------------------------------------------
    # 输出结构：所有必要的调试项与论文中的分支项
    # -----------------------------------------------------------
    parts = {
        # 时间一致性分支
        "temporal_score": temporal_score,  # 最终时间分支得分 = w_tdev * tdev_z + w_mag * mag_z
        "tdev_z": tdev_z,                 # 状态演化偏差（z-score）
        "mag_z": mag_z,                   # 状态幅值偏差（z-score）

        # 能量自然性分支
        "energy_score":   r_z + kappa_z,  # 保持原有写法
        "geom_score":     kms_z,          # 轨迹几何自然性分支
        "r_z": r_z,                       # 能量趋势偏差
        "kappa_z": kappa_z,               # 残差变化率偏差
        "kms_z": kms_z,                   # 多尺度轨迹自然性偏差

        # 秩融合相关
        "rank_T": rank_T,
        "rank_E": rank_E,
        "rank_avg": rank_avg,
    }

    return final.view(-1), parts

