# loss.py
# 统一能量–时间训练目标（EGU-AD）
#
# 结构对应两大模块：
# 1) Temporal Consistency Learning（时间一致性学习） → L_state
# 2) Energy-Driven Representation Learning（能量驱动表示学习） → L_flow, L_diss, L_geo
#
# 其中能量分支内部三项可解释为：
# - L_flow : Energy Trend Consistency（能量趋势一致性）
# - L_diss : Energy Residual Stability（能量残差稳定性）
# - L_geo  : Adaptive Energy Trajectory Smoothness（自适应能量轨迹平滑）
#
# 注意：
# - lambda_ctr / lambda_time 逻辑上已经废弃（恒为 0），仅为兼容 solver 的打印与参数传递。
# - 若 model.pgw 存在，则仍保留一个极小的先验正则 L_wprior（不影响主要行为，可视为高级选项）。


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
    """
    统一损失函数。

    Args
    ----
    model : nn.Module
        EGU-AD 主模型（可能包含 pgw 等子模块）。
    h : Tensor [T, D_h]
        几何分支 / 编码器输出的中间特征（作为对齐目标）。
    p : Tensor [T, D_p]
        最终用于异常检测的时间状态表示。
    h_targ_n : Tensor [T, D_p]
        h 的归一化版本或其它对齐目标（目前仅用于兼容项）。
    edge_index : Tensor
        图结构索引（在当前损失中未直接使用，保留以便扩展）。
    E : Tensor [T] 或 None
        能量标量轨迹。
    dE_pred : Tensor [T] 或 None
        预测的能量增量序列。

    Returns
    -------
    total_loss : Tensor (scalar)
        用于 backward 的总损失。
    loss_items : dict[str, float]
        各子项的标量值（已 detach），用于日志打印。
    parts : dict[str, Tensor]
        中间 Tensor，用于可视化 / debug。
    """
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

        # 1.1 能量趋势一致性：预测增量对齐真实增量
        #     对应：Energy Trend Consistency
        L_flow_t = F.mse_loss(dE_pred[1:], dE_true)

        # 1.2 自适应轨迹平滑的换挡遮罩：
        #     使用 |ΔE_true| 的 80% 分位数区分“低速正常区”和“模式切换区”
        base = dE_true.abs().detach().cpu().numpy().reshape(-1)
        q80 = float(np.percentile(base, 80.0)) if base.size > 0 else 0.0
        mask = (dE_true.abs() <= q80).float().to(device)  # [T-1] 低速段 = 更应平滑

        # 1.3 残差 r_t 及其时间平滑（TV），只在低速段约束
        #     对应：Residual Energy Stability 的一部分 + 轨迹平滑的一部分
        r_t_full = (dE_true - dE_pred[1:]).abs()  # [T-1]
        if r_t_full.numel() > 1:
            r_tv_all = F.mse_loss(
                r_t_full[1:], r_t_full[:-1], reduction="none"
            )                                    # [T-2]
            denom = mask[1:].sum().clamp_min(1.0)
            r_tv_t = (r_tv_all * mask[1:]).sum() / denom
        else:
            r_tv_t = _zero()

        # 1.4 能量轨迹的二阶差分（曲线形状），同样只在低速段约束
        #     对应：Adaptive Energy Trajectory Smoothness 的主体
        if E.numel() > 2:
            ddE = E[2:] - 2.0 * E[1:-1] + E[:-2]   # [T-2]
            L_gbe_all = ddE ** 2
            mask_ddE = mask[1:]                   # 对齐长度 [T-2]
            denom = mask_ddE.sum().clamp_min(1.0)
            L_gbe_t = (L_gbe_all * mask_ddE).sum() / denom
        else:
            L_gbe_t = _zero()

        # 1.5 残差能量的稳定性 + 可选 sigma 锚
        #     对应：Energy Residual Stability 的主体
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
        # 若模型定义了 pgw，则可以返回一个很小的先验正则
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
    # 6) 标量化日志输出
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

