import os, sys
import argparse
import types
import torch
import time

############################################################
# 0. 兼容补丁：拦截 torch._dynamo / torch.onnx 中的问题接口
############################################################

os.environ["PYTORCH_DISABLE_DYNAMO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"


import random, numpy as np

def set_seed(seed: int = 2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 完全确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # CUDA 11+ 需要
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# 有些 PyTorch 2.x 环境下，在禁用 dynamo/compile 时，
# torch._dynamo 会变成一个 SimpleNamespace，没有 disable()。
# 后面 optimizer 初始化路径里的 torch/_compile.py 会假设 disable() 存在，
# 于是报 AttributeError。我们在这里兜底补一个 no-op 的 disable。

if hasattr(torch, "_dynamo"):
    if isinstance(torch._dynamo, types.SimpleNamespace):
        def _no_op_disable(fn=None, recursive=None):
            # 允许两种调用姿势：
            # 1) torch._dynamo.disable()(f)
            # 2) torch._dynamo.disable(f, recursive=True/False)
            if fn is None:
                def decorator(inner_fn):
                    return inner_fn
                return decorator
            return fn
        torch._dynamo.disable = _no_op_disable


# 屏蔽 onnx exporter 相关接口
if hasattr(torch, "onnx"):
    # 1) fake torch.onnx.operators (有的老版本才有, 新版可能拆掉了)
    if not hasattr(torch.onnx, "operators"):
        torch.onnx.operators = types.SimpleNamespace()

    # 2) fake torch.onnx._internal.exporter and DiagnosticOptions
    if not hasattr(torch.onnx, "_internal"):
        torch.onnx._internal = types.SimpleNamespace()

    if not hasattr(torch.onnx._internal, "exporter"):
        torch.onnx._internal.exporter = types.SimpleNamespace()

    exp = torch.onnx._internal.exporter
    if not hasattr(exp, "DiagnosticOptions"):
        # 定义一个最小壳，防止 "from torch.onnx._internal.exporter import DiagnosticOptions" 失败
        class _FakeDiagnosticOptions:
            pass
        exp.DiagnosticOptions = _FakeDiagnosticOptions

############################################################
# 1. 继续正常 import 其余依赖（现在 torch 已被打补丁）
############################################################

import numpy as np
import random

from solver.solver_eie import SolverEIE


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)
    def flush(self):
        pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='PSM')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_save_path', type=str, default='cpt_eie')
    # parser.add_argument('--lambda_dev', type=float, default=0.05)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--win_size', type=int, default=110)
    parser.add_argument('--input_c', type=int, default=26)
    parser.add_argument('--d_model', type=int, default=512)  # hidden dim
    parser.add_argument('--e_layers', type=int, default=2)

    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--anormly_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=2)

    args = parser.parse_args()
    return args


def main(args):
    # 0. set seed
    set_seed(args.seed)

    # 1. init solver
    solver = SolverEIE(args)

    # 2. branch by mode
    if args.mode == 'train':
        solver.train()
        # 训练完自动测
        solver.test()
    elif args.mode == 'test':
        solver.test()
    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    # 强制关闭 dynamo trace (双保险)
    os.environ["TORCH_DISABLE_DYNAMO"] = "1"
    
    args = build_args()
    
    os.makedirs("logs", exist_ok=True)

    # 自动生成日志文件名，比如包含时间戳、数据集名
    log_filename = f"logs/{args.dataset}_{args.mode}_{time.strftime('%Y%m%d-%H%M%S')}.log"

    # 将标准输出流重定向
    sys.stdout = Logger(log_filename, add_flag=True)
    sys.stderr = Logger(log_filename, add_flag=True)

    main(args)
