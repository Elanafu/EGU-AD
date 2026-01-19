import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from data.data_loader import get_graph_loader
from model.eie_model import EIEModel
from model.loss import compute_loss
from model.anomaly_scoring import compute_anomaly_scores
from metrics.metrics import combine_all_evaluation_scores
import h5py


import random
from torch.backends import cudnn

def _zscore_1d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    简单的一维 z-score 标准化，用于构造 distill 的 teacher / student 信号。
    """
    if x.numel() == 0:
        return x
    m = x.mean()
    s = x.std(unbiased=False) + eps
    return (x - m) / s



def _move_batch_to_device(batch, device):
    """
    dataloader 返回 (data, labels)
    data: PyG Data/Batch，包含 .x (T,C), .edge_index (2,E), ...
    labels: (T,) tensor 或 numpy，可能为 None
    """
    if isinstance(batch, (tuple, list)):
        data, labels = batch
    else:
        data, labels = batch, None

    data = data.to(device)
    if labels is not None and torch.is_tensor(labels):
        labels = labels.to(device)

    return data, labels


class SolverEIE(object):
    def __init__(self, config):
        # ======================================================
        # 0) 解析 config -> self
        # ======================================================
        if isinstance(config, dict):
            cfg_dict = config
        else:
            cfg_dict = vars(config)

        for k, v in cfg_dict.items():
            setattr(self, k, v)

        # ======================================================
        # 1) dataloaders
        # ======================================================
        self.train_loader = get_graph_loader(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='train',
            dataset=self.dataset
        )
        self.vali_loader = get_graph_loader(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='val',
            dataset=self.dataset
        )
        self.thre_loader = get_graph_loader(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='thre',
            dataset=self.dataset
        )
        self.test_loader = get_graph_loader(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='test',
            dataset=self.dataset
        )

        # ======================================================
        # 2) device
        # ======================================================
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu}")
        else:
            self.device = torch.device("cpu")

        # ======================================================
        # 3) 模型 (注意：这里不再传 momentum)
        # ======================================================
        self.model = EIEModel(
            input_dim=self.input_c,
            hidden_dim=self.d_model,
            num_layers=self.e_layers,
        ).to(self.device)

        # ======================================================
        # 4) optimizer
        # ======================================================
        os.environ["TORCH_DISABLE_DYNAMO"] = "1"
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Info] Total parameters: {total_params}")

        # ======================================================
        # 5) loss 权重, 阈值策略
        # ======================================================
        self.lamb = dict(
            ctr=1.0,
            time=1.0,
            var=0.1,
        )

        self.threshold_mode = 'percentile'
        self.percentile_alpha = max(0.3, getattr(self, 'anormly_ratio', 1.0))
        self.th_k = 2.0

    def _forward_model(self, data_batch):
        out = self.model(data_batch)

        if not isinstance(out, (tuple, list)):
            raise RuntimeError("EIEModel.forward() must return tuple/list")

        if len(out) == 4:
            # 新模型接口: (h, p, E, dE_pred)
            h, p, E, dE_pred = out
            # 旧代码里的 h_targ_n 这里我们先定义成基于 h 的 teacher 表征
            # 方便后面 compute_loss 还能跑
            h_targ_n = F.normalize(h.detach(), p=2, dim=1)

        elif len(out) == 3:
            # 老形式: (h, p, h_targ_n)
            h, p, h_targ_n = out
            E = None
            dE_pred = None

        elif len(out) == 2:
            # 老老形式: (h, p)
            h, p = out
            h_targ_n = F.normalize(h.detach(), p=2, dim=1)
            E = None
            dE_pred = None

        else:
            raise RuntimeError(
                "EIEModel.forward() returned unexpected tuple length "
                f"{len(out)}; expected 2, 3, or 4."
            )

        # 现在返回五个量，后面train()会一起接
        return h, p, h_targ_n, E, dE_pred


    # ======================================================
    # validation loss per epoch
    # ======================================================
    @torch.no_grad()
    def _vali_epoch_loss(self):
        self.model.eval()

        total_list = []

        for raw_batch in self.vali_loader:
            # 把 batch 放到正确的 device
            data_batch, _ = _move_batch_to_device(raw_batch, self.device)

            # 前向
            h, p, h_targ_n, E, dE_pred = self._forward_model(data_batch)
            
            # 新（our8.0）：
            base_loss, loss_items = compute_loss(
                model=self.model,
                h=h, p=p, h_targ_n=h_targ_n, edge_index=data_batch.edge_index,
                E=E, dE_pred=dE_pred,
                # —— 物理损失：仍以能量守恒（耗散/几何）为主 ——
                lambda_diss=0.5,
                lambda_geo=0.2,
                # —— 极轻“防塌”稳定器（不改变异常定义，只保证可训练） ——
                lambda_state=1e-3,     # L2(p) 级别的小正则，防全零
                lambda_ctr=0.1,        # 很小的表征对齐，保证 p 不飘
                lambda_time=0.05,      # 很小的时间平滑，抑制训练噪声
                huber_delta=1.0,
                target_sigma=None
            )



            # flow_loss：和训练时一样的能量流动项
            if (E is not None) and (dE_pred is not None):
                if E.shape[0] > 1:
                    dE_true = torch.zeros_like(E)
                    dE_true[1:] = E[1:] - E[:-1]
                    flow_loss = torch.mean(torch.abs(dE_pred - dE_true))
                else:
                    flow_loss = torch.tensor(0.0, device=E.device)
            else:
                flow_loss = torch.tensor(0.0, device=h.device if hasattr(h, "device") else "cpu")

            total_loss = base_loss + 0.1 * flow_loss

            total_list.append(total_loss.item())

        # 返回验证集平均loss
        return float(np.mean(total_list))
    
    @torch.no_grad()
    def _fit_geom_gate(self):
        """
        基于 thre_loader（无标签）做一次统计预估，决定几何分支的全局门控强度。
        原则（全程无监督）：
        - alignment:  corr = corr(S, G) 的正向程度（负相关时不启用）
        - tail_gain:  G 在 S 的高分尾部是否显著更高（利用 95 分位比较）
        - stability:  控制 G 的过度方差，抑制抖动（利用 MAD）
        最终：beta_adapt = beta0 * sqrt(max(0,corr)) * relu(tail_gain) / (1 + jitter)
        同时设置 use_geom_adapt = beta_adapt > 极小阈值
        """
        self.model.eval()

        # 可调起点（超参只做缩放，不绑定数据集）
        beta0 = float(getattr(self, "beta_geom", 0.3))      # 全局上限（建议 0.2~0.5 起步）
        q_hi  = float(getattr(self, "gate_q_high", 0.95))   # S 的高分尾部分位
        q_lo  = float(getattr(self, "gate_q_low", 0.50))    # 对比：中位附近
        eps   = 1e-6

        S_all = []
        G_all = []

        for raw_batch in self.thre_loader:
            data_batch, _ = _move_batch_to_device(raw_batch, self.device)

            # ===== 计算 v8 时间分支分数 S（与当前推理完全一致）=====
            h, p, h_targ_n, E, dE_pred = self._forward_model(data_batch)

            # v8分数
            base_score, _ = compute_anomaly_scores(h, p)
            E_baseline = E.mean()
            energy_term = torch.relu(E - E_baseline)
            flow_term   = torch.abs(dE_pred)
            raw_score   = base_score + 0.5*flow_term + 0.5*energy_term
            r = int(getattr(self, "seg_radius", 3))
            kernel  = torch.ones(1,1,2*r+1, device=raw_score.device) / (2*r+1)
            support = F.conv1d(raw_score.view(1,1,-1), kernel, padding=r).view(-1)
            segment_score = raw_score * support
            spike_score   = raw_score
            spike_z   = (spike_score   - spike_score.mean())   / (spike_score.std()   + eps)
            segment_z = (segment_score - segment_score.mean()) / (segment_score.std() + eps)
            lambda_mix = float(getattr(self, "lambda_mix", 0.3))
            S = torch.max(spike_z, segment_z) + lambda_mix * 0.5*(spike_z + segment_z)  # (T,)

            # ===== 计算几何分支原始 κ_ms 并 z 标准化，得到 G =====
            def _ensure_odd(k): return k if (k % 2 == 1) else (k + 1)
            def _med_lenT(x, k):
                k = _ensure_odd(int(k))
                if k <= 1: return x
                left, right = k//2, k-1-k//2
                x_pad = F.pad(x.view(1,1,-1), (left, right), mode='reflect').view(-1)
                return x_pad.unfold(0, k, 1).median(dim=-1).values
            def _mean_lenT(x,k):
                k = _ensure_odd(int(k))
                if k <= 1: return x
                left, right = k//2, k-1-k//2
                w = torch.ones(1,1,k, device=x.device, dtype=x.dtype)/k
                return F.conv1d(x.view(1,1,-1).pad((left,right), "reflect"), w).view(-1)

            # 多尺度（统一使用与推理一致的缺省，避免不一致）
            scales  = tuple(getattr(self, "geom_scales", (1,4,16)))
            k_med   = int(getattr(self, "geom_k_med", 5))
            k_mean  = int(getattr(self, "geom_k_mean", 5))

            outs = []
            for s in scales:
                ks = max(1, int(s * k_med))
                km = max(1, int(s * k_mean))
                Em = _med_lenT(E, ks)
                Es = _mean_lenT(Em, km)

                d1 = torch.zeros_like(Es)
                if Es.numel() > 1:
                    d1[1:-1] = 0.5*(Es[2:] - Es[:-2])
                    d1[0]    = Es[1]-Es[0]
                    d1[-1]   = Es[-1]-Es[-2]

                d2 = torch.zeros_like(d1)
                if d1.numel() > 1:
                    d2[1:-1] = torch.abs(d1[2:] - d1[:-2])
                    d2[0]    = torch.abs(d1[1]-d1[0])
                    d2[-1]   = torch.abs(d1[-1]-d1[-2])

                # 裁剪 + 压缩
                p_hi = torch.quantile(d2, 0.99)
                p_lo = torch.quantile(d2, 0.01)
                d2   = torch.clamp(d2, min=p_lo, max=p_hi)
                q95  = torch.quantile(d2, 0.95)
                d2n  = torch.tanh(d2/(q95+eps))
                outs.append(d2n)

            kappa_ms = torch.stack(outs, dim=0).max(dim=0).values
            G = (kappa_ms - kappa_ms.mean()) / (kappa_ms.std() + eps)

            S_all.append(S.detach().cpu())
            G_all.append(G.detach().cpu())

        S_all = torch.cat(S_all, dim=0)
        G_all = torch.cat(G_all, dim=0)

        # 对齐度（相关）
        Sa = S_all - S_all.mean(); Ga = G_all - G_all.mean()
        corr = float(((Sa*Ga).mean() / ((Sa.std()+eps)*(Ga.std()+eps))).clamp(-1,1).item())

        # 尾部分离度：G 在 S 的高分尾（≥ q_hi）是否显著更高于中位附近（≤ q_lo）
        s_hi = torch.quantile(S_all, q_hi)
        s_lo = torch.quantile(S_all, q_lo)
        G_hi = G_all[S_all >= s_hi]
        G_lo = G_all[S_all <= s_lo]
        tail_gain = float((G_hi.mean() - G_lo.mean()).item()) if (G_hi.numel()>5 and G_lo.numel()>5) else 0.0
        tail_gain = max(0.0, tail_gain)  # 只取正增益

        # 稳定性：G 在高分尾部的 “抖动” 惩罚（MAD）
        if G_hi.numel() > 5:
            med_hi = G_hi.median()
            mad_hi = (G_hi - med_hi).abs().median().item()
        else:
            mad_hi = 0.0
        jitter = mad_hi  # 抖动越大，权重越小

        # 综合门控（无监督）
        #   - 负相关：直接关闭（≈0）
        #   - 正相关：beta0 * sqrt(corr) * relu(tail_gain) / (1 + jitter)
        if corr <= 0:
            beta_adapt = 0.0
        else:
            beta_adapt = beta0 * (corr ** 0.5) * (tail_gain) / (1.0 + jitter + 1e-6)

        # 软阈值：太小就当关闭
        if beta_adapt < 1e-3:
            self.use_geom_adapt = False
            self.beta_adapt = 0.0
        else:
            self.use_geom_adapt = True
            # 上限保护，避免过大
            self.beta_adapt = float(min(beta_adapt, beta0))

        return {"corr": corr, "tail_gain": tail_gain, "jitter": jitter, "beta_adapt": self.beta_adapt}

    
        
    # ====================== 自适应权重_v6======================"
    def train(self):
        print("====================== TRAIN MODE ======================")
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # ---- 强确定性（训练期）----
        strict_det = bool(getattr(self, "strict_deterministic", True))
        if strict_det:
            seed = int(getattr(self, "seed", 42))

            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # 让 cublas 可重复
            os.environ["PYTHONHASHSEED"] = str(seed)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # 关键：遇到非确定性算子直接抛错（别 warn_only）
            torch.use_deterministic_algorithms(True)

            cudnn.deterministic = True
            cudnn.benchmark = False


        # —— 自适应多尺度的正则权重（缺省更安全）——
        lambda_gate_ent = getattr(self, "lambda_gate_ent", 1e-3)  # 熵正则
        lambda_gate_bal = getattr(self, "lambda_gate_bal", 1e-3)  # 分支均衡
        # 可选梯度裁剪
        grad_clip = float(getattr(self, "grad_clip", 0.0))        # 0 表示不裁剪
        self._p_ema = None
        ema_alpha = float(getattr(self, "p_ema_alpha", 0.9))

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_time = time.time()

            total_list = []
            ctr_list   = []
            time_list  = []
            var_list   = []
            flow_list  = []
            gate_ent_list = []   # 记录门控熵正则
            gate_bal_list = []   # 记录门控均衡正则

            # 新增：按 epoch 聚合的几项
            diss_list = []
            geo_list  = []

            pbar = tqdm(self.train_loader, total=len(self.train_loader), ncols=120)

            for raw_batch in pbar:
                # 1. data
                data_batch, _ = _move_batch_to_device(raw_batch, self.device)

                # 2. forward
                self.optimizer.zero_grad(set_to_none=True)
                h, p, h_targ_n, E, dE_pred = self._forward_model(data_batch)
                
                
                # 2) 维护 stop-grad EMA 作为对齐目标
                if self._p_ema is None or (self._p_ema.shape != p.shape):
                    self._p_ema = p.detach().clone()
                else:
                    self._p_ema.mul_(ema_alpha).add_((1.0 - ema_alpha) * p.detach())

                # 3) 轻时间稳定项调度（热身->稳定）
                #    例如: 前 20% epoch 线性升到设定值，其后余弦小幅波动
                base_lambda_state = float(getattr(self, "lambda_state_base", 1e-2))
                warm_frac = 0.2
                progress = (epoch + 1) / self.num_epochs
                if progress < warm_frac:
                    lambda_state_now = base_lambda_state * (progress / warm_frac)
                else:
                    # 余弦扰动（可选，不想要就直接 = base_lambda_state）
                    import math
                    lambda_state_now = base_lambda_state * (0.75 + 0.25 * (1 + math.cos(math.pi * (progress - warm_frac)/(1-warm_frac))))



                # 4) 计算损失（把 p_ema 传进去）
                base_loss_raw, loss_items, parts = compute_loss(
                    model=self.model,
                    h=h, p=p, h_targ_n=h_targ_n, edge_index=data_batch.edge_index,
                    E=E, dE_pred=dE_pred,
                    lambda_diss=0.5, lambda_geo=0.2,
                    lambda_state=lambda_state_now,    # ★ 用“当前值”
                    lambda_ctr=0.0, lambda_time=0.0,
                    huber_delta=1.0, target_sigma=None,
                    p_ema=self._p_ema                # ★ 关键
                )
                
                
                # === 新增：EMA 标尺，动态归一化 ===
                if not hasattr(self, "_ema_loss_scale"):
                    self._ema_loss_scale = {k: 1.0 for k in [
                        "L_flow","L_diss","L_geo","L_state","L_ctr","L_time",
                        "L_r_smooth","L_gbe","L_sigma"
                    ]}


                # 更新 EMA（0.9 平滑）
                for k in self._ema_loss_scale.keys():
                    if k in loss_items:
                        v = max(1e-8, float(loss_items[k]))
                        self._ema_loss_scale[k] = 0.9*self._ema_loss_scale[k] + 0.1*v

                def _n_t(k):  # 归一化后的张量
                    key = f"{k}_t"
                    s = self._ema_loss_scale.get(k, 1.0)
                    return parts[key] / (s + 1e-8)

                # === 用归一化张量重新组装 total_loss ===
                total_loss = (
                    1.0*_n_t("L_flow")      # 守恒对齐（主力）
                + 1.0*_n_t("L_diss")      # 耗散稳健项（Huber + 可选 sigma 锚）
                + 0.6*_n_t("L_geo")       # 几何：弯曲能 + 残差TV
                + 0.2*_n_t("L_state")     # 轻状态稳定（数值）
                + 0.1*_n_t("L_time")      # 兼容的小时间平滑（如需可改 0）
                + 0.1*_n_t("L_ctr")       # 兼容的对齐（如需可改 0）
                )

                # 3c. 门控正则（保持你的原逻辑）
                L_gate_ent = torch.tensor(0.0, device=h.device)
                L_gate_bal = torch.tensor(0.0, device=h.device)
                try:
                    alpha_tb = getattr(getattr(self.model, "encoder", None), "temporal_ms", None)
                    if alpha_tb is not None and hasattr(alpha_tb, "last_alpha") and alpha_tb.last_alpha is not None:
                        a = alpha_tb.last_alpha.detach().clamp(1e-8, 1.0)
                        H = -(a * a.log()).sum(dim=-1).mean()
                        L_gate_ent = 0.1 * H
                        mean_b = a.mean(dim=0)
                        target = torch.full_like(mean_b, 1.0 / mean_b.shape[0])
                        L_gate_bal = F.mse_loss(mean_b, target)
                except Exception:
                    pass

                # 3d. 总损失
                total_loss = total_loss + lambda_gate_ent * L_gate_ent + lambda_gate_bal * L_gate_bal

                # 4. backward
                total_loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

                # 5. log accum（兼容不同版本的 loss_items：缺失键按 0 处理）
                def _li(d, k, default=0.0):
                    v = d.get(k, default)
                    try:
                        return float(v.item() if hasattr(v, "item") else v)
                    except Exception:
                        return float(default)

                L_ctr        = _li(loss_items, "L_ctr", 0.0)
                L_time       = _li(loss_items, "L_time", 0.0)
                L_var        = _li(loss_items, "L_var", 0.0)
                L_flow       = _li(loss_items, "L_flow", 0.0)
                L_diss       = _li(loss_items, "L_diss", 0.0)
                L_geo        = _li(loss_items, "L_geo", 0.0)

                total_list.append(float(total_loss.item()))
                ctr_list.append(L_ctr)
                time_list.append(L_time)
                var_list.append(L_var)
                flow_list.append(L_flow)
                diss_list.append(L_diss)
                geo_list.append(L_geo)
                gate_ent_list.append(float(L_gate_ent.item()))
                gate_bal_list.append(float(L_gate_bal.item()))

                # 进度条
                pbar.set_description(
                    f"Epoch {epoch+1}/{self.num_epochs} | "
                    f"Loss {total_loss.item():.4f} | "
                    f"flow {L_flow:.3f} | diss {L_diss:.3f} | "
                    f"geo {L_geo:.3f} | state {loss_items['L_state']:.3f} | "
                    f"time {loss_items['L_time']:.3f} | ctr {loss_items['L_ctr']:.3f} | "
                    f"rSm {loss_items['L_r_smooth']:.3f} | gbe {loss_items['L_gbe']:.3f}"
                )

            # ====== 每个 epoch 结束后：验证 ======
            with torch.no_grad():
                self.model.eval()
                vali_losses = []
                for raw_batch in self.vali_loader:
                    data_batch, _ = _move_batch_to_device(raw_batch, self.device)
                    h, p, h_targ_n, E, dE_pred = self._forward_model(data_batch)
                    loss_val, _, _ = compute_loss(
                        model=self.model,
                        h=h, p=p, h_targ_n=h_targ_n, edge_index=data_batch.edge_index,
                        E=E, dE_pred=dE_pred,
                        lambda_diss=0.5, lambda_geo=0.2, lambda_state=1e-3,
                        lambda_ctr=0.0, lambda_time=0.0,
                        huber_delta=1.0, target_sigma=None
                    )
                    vali_losses.append(float(loss_val.item()))
                vali_loss = float(sum(vali_losses) / max(1, len(vali_losses)))

            # === 打印 ===
            print(
                f"[Epoch {epoch+1}] "
                f"TrainLoss={np.mean(total_list):.6f}  "
                f"ValLoss={vali_loss:.6f}  "
                f"L_flow={np.mean(flow_list):.4f}  "
                f"L_diss={np.mean(diss_list):.4f}  "
                f"L_geo={np.mean(geo_list):.4f}  "
                f"L_gateEnt={np.mean(gate_ent_list):.4f}  "
                f"L_gateBal={np.mean(gate_bal_list):.4f}  "
                f"Time={time.time()-epoch_time:.2f}s"
            )

            # ====== 保存 ======
            ckpt = {
                "model": self.model.state_dict(),
                "opt": self.optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(
                ckpt,
                os.path.join(self.model_save_path, f"{self.dataset}_checkpoint.pth")
            )
    

    


#  ======================自适应权重_v6（先验+自适应）======================
    @torch.no_grad()
    def _infer_scores_on_batch(
        self,
        data_batch,
        return_parts: bool = False,
        w_rank_EG: float = None,
        calib: dict = None
    ):
        """
        推理时的统一入口：
        - 只在 w_rank_EG 上支持自适应（dataset 级别），默认用 self._w_rank_EG 或配置。
        - EG 内部 wR/wK/wMS 使用固定的物理先验比例：0.50 / 0.25 / 0.25。
        """
        self.model.eval()
        h, p, h_targ_n, E, dE_pred = self._forward_model(data_batch)

        # 1) 是否启用能量分支（保持你原来的开关）
        use_energy_flag = bool(getattr(self, 'use_energy_eval', True))

        # 2) 晚融合权重 w_rank_EG：
        #    优先使用函数入参（校准阶段会显式传入），否则用 self._w_rank_EG 或默认 0.7
        if w_rank_EG is not None:
            w_rank = float(w_rank_EG)
        else:
            w_rank = float(getattr(self, "_w_rank_EG", getattr(self, "w_rank_EG_eval", 0.75)))

        # 3) EG 内部固定权重（论文中可以写成“物理先验下的均衡线性组合”）
        wR  = float(getattr(self, "wR_eval", 0.50))
        wK  = float(getattr(self, "wK_eval", 0.25))
        wMS = float(getattr(self, "wMS_eval", 0.25))

        # 4) 时间分支内部权重（保持现在这套配置）
        w_temporal_dev = float(getattr(self, "w_temporal_dev_eval", 0.75))
        w_temporal_mag = float(getattr(self, "w_temporal_mag_eval", 0.25))

        score, parts = compute_anomaly_scores(
            h, p, E=E, dE_pred=dE_pred,
            use_energy=use_energy_flag,
            w_temporal_dev=w_temporal_dev,
            w_temporal_mag=w_temporal_mag,
            wR=wR, wK=wK, wMS=wMS,
            w_rank_EG=w_rank,
        )

        # Debug 打印（现在 wR/wK/wMS 固定，w_rank_EG 可能由 test() 校准）
        try:
            print(
                f"[Debug] eg_mode={parts.get('eg_mode')} | "
                f"wR={parts.get('wR'):.2f}, wK={parts.get('wK'):.2f}, wMS={parts.get('wMS'):.2f} | "
                f"w_rank_EG={w_rank:.2f}"
            )
        except Exception:
            pass

        if return_parts:
            return score, parts
        return score


    @torch.no_grad()
    def test(self):
        print("======================TEST MODE======================")
        ckpt_path = os.path.join(str(self.model_save_path), f"{self.dataset}_checkpoint.pth")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict_to_load = ckpt["model"] if "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict_to_load, strict=False)
        self.model.eval()

        # ---------- 1) 在 thre split 上估计阈值（保持原逻辑） ----------
        thre_scores = []
        for raw_batch in self.thre_loader:
            data_batch, _ = _move_batch_to_device(raw_batch, self.device)
            s = self._infer_scores_on_batch(data_batch)  # 不需要 parts
            thre_scores.append(s.detach().cpu().numpy())
        thre_scores = np.concatenate(thre_scores, axis=0).astype(np.float64).reshape(-1)

        alpha = float(getattr(self, "anormly_ratio", 1.0))
        q_low = float(getattr(self, "thre_q_low", 60.0))
        q_high = float(getattr(self, "thre_q_high", 99.0))
        q_thre = float(np.clip(100.0 - alpha, q_low, q_high))
        thresh = float(np.percentile(thre_scores, q_thre))
        print(f"[Info] THRE percentile (q={q_thre:.1f}) from thre split: thr={thresh:.6f}")

        # ---------- 2) 测试集打分 + 收集中间分数 ----------
        all_test_scores = []
        all_test_labels = []

        for raw_batch in self.test_loader:
            data_batch, labels = _move_batch_to_device(raw_batch, self.device)

            scores, parts = self._infer_scores_on_batch(data_batch, return_parts=True)

            # 最终用于评测 / 阈值的 score
            all_test_scores.append(scores.detach().cpu().numpy())

            # labels
            lab_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
            all_test_labels.append(lab_np)

        all_test_scores = np.concatenate(all_test_scores, axis=0).astype(np.float64).reshape(-1)
        all_test_labels = np.concatenate(all_test_labels, axis=0).reshape(-1).astype(int)

        pred_bin = (all_test_scores > thresh).astype(int)
        print(f"[Info] Binarize rate on TEST: {pred_bin.mean() * 100:.3f}%")
        gt = all_test_labels
        metrics = combine_all_evaluation_scores(pred_bin, gt)

        clean_metrics = {}
        for k, v in metrics.items():
            try:
                clean_metrics[k] = v.item() if hasattr(v, "item") else float(v)
            except Exception:
                clean_metrics[k] = v
        print("[Test Result]")
        print(clean_metrics)

        return clean_metrics