import torch
import torch.nn.functional as F
import numpy as np


def compute_loss(
    model,
    h,
    p,
    h_targ_n,
    edge_index,
    lamb=None,
    E=None,
    dE_pred=None,
    lambda_diss: float = 0.5,
    lambda_geo: float = 0.2,
    lambda_state: float = 0.1,   # 时间一致性权重（外层可做调度）
    lambda_ctr: float = 0.0,     # 兼容参数：现已逻辑废弃
    lambda_time: float = 0.0,    # 兼容参数：现已逻辑废弃
    huber_delta: float = 1.0,
    target_sigma: float | None = None,
    p_ema: torch.Tensor | None = None,   # EMA 目标，用于状态一致性（可选）
    lambda_w_prior: float = 1e-3,        # 若存在 model.pgw，则施加极小先验正则
    **kwargs,
):

    device = p.device
    eps = 1e-8

    def _zero():
        return torch.tensor(0.0, device=device)

    def _huber(x, delta: float):
        ax = x.abs()
        quad = torch.clamp(ax, max=delta)
        lin = ax - quad
        return (0.5 * quad ** 2 + delta * lin).mean()

    # ------------------------------------------------------------------
    # 1) 能量分支：Energy-Driven Representation Learning (EDR)
    #    - L_flow : Energy Trend Consistency
    #    - L_diss : Energy Residual Stability
    #    - L_geo  : Adaptive Energy Trajectory Smoothness
    # ------------------------------------------------------------------
    if (E is not None) and (dE_pred is not None) and (E.shape[0] > 1):
        E = E.view(-1)
        dE_pred = dE_pred.view(-1)

        # 真实能量增量 ΔE_true
        dE_true = E[1:] - E[:-1]                  # [T-1]


        L_flow_t = F.mse_loss(dE_pred[1:], dE_true)

        base = dE_true.abs().detach().cpu().numpy().reshape(-1)
        q80 = float(np.percentile(base, 80.0)) if base.size > 0 else 0.0
        mask = (dE_true.abs() <= q80).float().to(device)  # [T-1] 低速段 = 更应平滑

        r_t_full = (dE_true - dE_pred[1:]).abs()  # [T-1]
        if r_t_full.numel() > 1:
            r_tv_all = F.mse_loss(
                r_t_full[1:], r_t_full[:-1], reduction="none"
            )                                    # [T-2]
            denom = mask[1:].sum().clamp_min(1.0)
            r_tv_t = (r_tv_all * mask[1:]).sum() / denom
        else:
            r_tv_t = _zero()


        if E.numel() > 2:
            ddE = E[2:] - 2.0 * E[1:-1] + E[:-2]   # [T-2]
            L_gbe_all = ddE ** 2
            mask_ddE = mask[1:]                   # 对齐长度 [T-2]
            denom = mask_ddE.sum().clamp_min(1.0)
            L_gbe_t = (L_gbe_all * mask_ddE).sum() / denom
        else:
            L_gbe_t = _zero()


        L_diss_huber_t = _huber(r_t_full, huber_delta)
        if target_sigma is not None:
            sigma = torch.std(E, unbiased=False) + eps
            L_sigma_t = (sigma - float(target_sigma)) ** 2
        else:
            L_sigma_t = _zero()

        L_diss_t = L_diss_huber_t + L_sigma_t
        # 几何/轨迹项：二阶 + 残差 TV，统称为 L_geo
        L_geo_t = L_gbe_t + r_tv_t
    else:
        L_flow_t = _zero()
        L_diss_t = _zero()
        L_geo_t = _zero()
        r_tv_t = _zero()
        L_gbe_t = _zero()
        L_sigma_t = _zero()

    # ------------------------------------------------------------------
    # 2) 时间一致性：Temporal Consistency Learning (TCL)
    #    - L_state : 对 p 的时间/EMA 一致性约束
    # ------------------------------------------------------------------
    if lambda_state > 0 and p.shape[0] > 0:
        if (p_ema is not None) and (p_ema.shape == p.shape):
            # 与 EMA 目标对齐（stop-grad）
            L_state_t = F.mse_loss(p, p_ema.detach())
        elif p.shape[0] > 1:
            # 简化版时间平滑：鼓励相邻状态变化不过于剧烈
            dp = p[1:] - p[:-1]
            L_state_t = _huber(dp.norm(p=2, dim=-1), delta=1.0)
        else:
            L_state_t = _zero()
    else:
        L_state_t = _zero()

    # ------------------------------------------------------------------
    # 3) 兼容项（之前的可选项，现统一置零，仅保留字段，方便日志与兼容）
    # ------------------------------------------------------------------
    L_ctr_t = _zero()   # 之前的对齐项，不再使用
    L_time_t = _zero()  # 之前的额外时间平滑，不再使用

    # ------------------------------------------------------------------
    # 4) PGLW / 其他先验正则（若存在）
    # ------------------------------------------------------------------
    L_wprior_t = _zero()
    if (lambda_w_prior > 0.0) and hasattr(model, "pgw") and (model.pgw is not None):
        L_wprior_t = model.pgw.prior_loss(lambda_w=lambda_w_prior).to(device)

    # ------------------------------------------------------------------
    # 5) 汇总总损失
    # ------------------------------------------------------------------
    total_t = (
        L_flow_t
        + lambda_diss * L_diss_t
        + lambda_geo * L_geo_t
        + lambda_state * L_state_t
        # 兼容参数：虽然参与加权，但对应项恒为 0，因此不会改变数值
        + lambda_ctr * L_ctr_t
        + lambda_time * L_time_t
        + L_wprior_t
    )

    # ------------------------------------------------------------------
    # 6) 日志输出
    # ------------------------------------------------------------------
    loss_items = {
        "L_flow": float(L_flow_t.detach().item()),
        "L_diss": float(L_diss_t.detach().item()),
        "L_geo": float(L_geo_t.detach().item()),
        "L_state": float(L_state_t.detach().item()),
        "L_ctr": float(L_ctr_t.detach().item()),
        "L_time": float(L_time_t.detach().item()),
        "L_r_smooth": float(r_tv_t.detach().item()),
        "L_gbe": float(L_gbe_t.detach().item()),
        "L_sigma": float(L_sigma_t.detach().item()),
        "L_wprior": float(L_wprior_t.detach().item()),
    }

    parts = {
        "L_flow_t": L_flow_t,
        "L_diss_t": L_diss_t,
        "L_geo_t": L_geo_t,
        "L_state_t": L_state_t,
        "L_ctr_t": L_ctr_t,
        "L_time_t": L_time_t,
        "L_r_smooth_t": r_tv_t,
        "L_gbe_t": L_gbe_t,
        "L_sigma_t": L_sigma_t,
        "L_wprior_t": L_wprior_t,
    }

    return total_t, loss_items, parts

