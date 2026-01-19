import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import torch_geometric
from torch_geometric.data import Batch


class GraphPSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/PSM_train.csv')
        # data = np.load(data_path +'/WADI_train.npy')

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/PSM_test.csv')
        # test_data = np.load(data_path + '/WADI_test.npy')

        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]


        self.test_labels = pd.read_csv(data_path + '/PSM_test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of samples in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1
        
    def __getitem__(self, index):
       
        index = index * self.step
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        else:
            data =self.val[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index,label=label)
        
        return graph, torch.tensor(label)




class GraphMSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", add_self_loop: bool = False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.add_self_loop = bool(add_self_loop)

        # 标准化
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data[: int(len(data) * 0.8)]
        self.val   = data[int(len(data) * 0.8) :]
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")

        # —— 预构一份“邻接链”边模板（双向 ±1），复用以提速 —— #
        self.edge_index_tmpl = self._build_chain_graph(self.win_size, self.add_self_loop)

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    @staticmethod
    def _build_chain_graph(T: int, add_self_loop: bool = False) -> torch.Tensor:
        """构造长度为 T 的时间邻接链（双向），可选自环。返回 [2, E] 的 edge_index。"""
        if T <= 1:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            if add_self_loop and T == 1:
                loop = torch.tensor([[0],[0]], dtype=torch.long)
                edge_index = loop
            return edge_index

        src = np.arange(0, T - 1, dtype=np.int64)
        dst = src + 1
        # 双向边拼接
        e0 = np.stack([src, dst], axis=0)      # i -> i+1
        e1 = np.stack([dst, src], axis=0)      # i+1 -> i
        edges = np.concatenate([e0, e1], axis=1)  # [2, 2*(T-1)]

        edge_index = torch.tensor(edges, dtype=torch.long)
        if add_self_loop:
            loop = torch.arange(T, dtype=torch.long).unsqueeze(0).repeat(2, 1)  # [2, T]
            edge_index = torch.cat([edge_index, loop], dim=1)
        return edge_index.contiguous()

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            # 测试集按整窗滑动（步长=win_size），保持与你原逻辑一致
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step

        if self.mode == "train":
            data  = self.train[index : index + self.win_size]
            label = self.test_labels[index : index + self.win_size]
        elif self.mode == "val":
            data  = self.val[index : index + self.win_size]
            label = self.test_labels[index : index + self.win_size]
        elif self.mode == "test":
            data  = self.test[index : index + self.win_size]
            label = self.test_labels[index : index + self.win_size]
        else:
            base  = (index // self.step) * self.win_size
            data  = self.val[base : base + self.win_size]
            label = self.test_labels[base : base + self.win_size]

        # 节点特征（每个时间步一节点，特征=全部传感器通道）
        x = torch.tensor(data, dtype=torch.float32)             # [T, C]
        edge_index = self.edge_index_tmpl                       # 复用模板，避免重复构图

        graph = Data(x=x, edge_index=edge_index, label=torch.tensor(label))
        return graph, torch.tensor(label)


class GraphSMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # 读并标准化
        data = np.load(data_path + "/SMAP_train.npy").astype(np.float32)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = np.load(data_path + "/SMAP_test.npy").astype(np.float32)
        self.test = self.scaler.transform(test_data)

        self.train = data[: int(len(data) * 0.8)]
        self.val   = data[int(len(data) * 0.8):]

        # ---- 读标签并统一为 1D 二值序列 ----
        y = np.load(data_path + "/SMAP_test_label.npy", allow_pickle=True)
        y = np.asarray(y)
        if y.ndim == 1:
            y1d = (y > 0.5).astype(np.float32)
        else:
            # 多列标签：任一维异常→该时刻异常
            y1d = (y > 0.5).any(axis=1).astype(np.float32)
        if len(y1d) != len(self.test):
            raise ValueError(f"[SMAP] test rows {len(self.test)} != label rows {len(y1d)}")
        self.test_labels_1d = y1d  # [N_test,]

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("[SMAP] positives in test labels:", int(self.test_labels_1d.sum()))

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            # 非重叠窗口：步长=win_size
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step

        if self.mode == "train":
            data = self.train[index : index + self.win_size]
            label = np.zeros((self.win_size,), dtype=np.float32)
        elif self.mode == "val":
            data = self.val[index : index + self.win_size]
            label = np.zeros((self.win_size,), dtype=np.float32)
        elif self.mode == "test":
            # 非重叠切窗（与 __len__ 一致）
            base  = (index // self.step) * self.win_size
            data  = self.test[base : base + self.win_size]
            label = self.test_labels_1d[base : base + self.win_size]  # 1D
        else:
            base  = (index // self.step) * self.win_size
            data  = self.val[base : base + self.win_size]
            label = np.zeros((self.win_size,), dtype=np.float32)

        # Graph data
        x = torch.tensor(data, dtype=torch.float32)

        # 时间邻接 i ↔ i±1
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i - 1), min(self.win_size, i + 2)):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        graph = Data(x=x, edge_index=edge_index)
        return graph, torch.tensor(label, dtype=torch.float32)

class GraphSMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # 读并标准化
        data = np.load(data_path + "/SMD_train.npy").astype(np.float32)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = np.load(data_path + "/SMD_test.npy").astype(np.float32)
        self.test = self.scaler.transform(test_data)

        self.train = data[: int(len(data) * 0.8)]
        self.val   = data[int(len(data) * 0.8):]

        # ---- 读标签并统一为 1D 二值序列（多列→按列 OR）----
        y = np.load(data_path + "/SMD_test_label.npy", allow_pickle=True)
        y = np.asarray(y)
        if y.ndim == 1:
            y1d = (y > 0.5).astype(np.float32)
        else:
            y1d = (y > 0.5).any(axis=1).astype(np.float32)
        if len(y1d) != len(self.test):
            raise ValueError(f"[SMD] test rows {len(self.test)} != label rows {len(y1d)}")
        self.test_labels_1d = y1d  # [N_test,]

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("[SMD] positives in test labels:", int(self.test_labels_1d.sum()))

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            # 非重叠窗口：步长=win_size
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step

        if self.mode == "train":
            data  = self.train[index : index + self.win_size]
            label = np.zeros((self.win_size,), dtype=np.float32)   # 占位
        elif self.mode == "val":
            data  = self.val[index : index + self.win_size]
            label = np.zeros((self.win_size,), dtype=np.float32)   # 占位
        elif self.mode == "test":
            base  = (index // self.step) * self.win_size           # 非重叠切窗
            data  = self.test[base : base + self.win_size]
            label = self.test_labels_1d[base : base + self.win_size]  # 1D
        else:
            base  = (index // self.step) * self.win_size
            data  = self.val[base : base + self.win_size]
            label = np.zeros((self.win_size,), dtype=np.float32)   # 占位

        # Graph（时间相邻 i↔i±1）
        x = torch.tensor(data, dtype=torch.float32)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i - 1), min(self.win_size, i + 2)):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        graph = Data(x=x, edge_index=edge_index)
        return graph, torch.tensor(label, dtype=torch.float32)


class GraphServiceSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(data_path + "_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        else:
            data = self.val[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size] 
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index)
        
        return graph, torch.tensor(label)
        
class GraphSWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SWaT_train.npy", allow_pickle=True)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SWaT_test.npy", allow_pickle=True)
        self.test = self.scaler.transform(test_data)
        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(data_path + "/SWaT_test_label.npy", allow_pickle=True)
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        else:
            data =self.val[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index,label=label)
        
        return graph, torch.tensor(label)
    
class GraphWADISegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/WADI_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/WADI_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/WADI_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        else:
            data = self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size] 
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

        
        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index,label=label)
        
        return graph, torch.tensor(label)

class GraphNIPS_TS_WaterSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Water_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Water_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Water_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        else:
            data = self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size] 
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]
        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-2), min(self.win_size, i+3)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index,label=label)
        
        return graph, torch.tensor(label)     

class GraphNIPS_TS_SwanSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Swan_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Swan_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'val'):
            data = self.val[index:index + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif (self.mode == 'test'):
            data = self.test[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        else:
            data = self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size] 
            label = self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

        
        # Create graph data
        x = torch.tensor(data, dtype=torch.float32)  # Node features (time steps)
        
        # Create edges based on temporal relations (i.e., connect each time step to its neighbors)
        edge_index = []
        for i in range(self.win_size):
            for j in range(max(0, i-1), min(self.win_size, i+2)):  # Connect neighboring time steps
                if i != j:
                    edge_index.append([i, j])  # Add edge between time step i and j
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Return as a graph (Data object)
        graph = Data(x=x, edge_index=edge_index,label=label)
        
        return graph, torch.tensor(label)


def custom_collate_fn(batch):
    data_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]
    batch_data = Batch.from_data_list(data_list)
    batch_labels = torch.stack(label_list)
    return batch_data, batch_labels


def get_graph_loader(data_path, batch_size, win_size=100, step=1, mode='train',dataset='KDD'):
    """
    Returns a DataLoader for graph-based time-series data.
    """
    # dataset = GraphTimeSeriesLoader(data_path, win_size, step, mode)
    if (dataset == 'SMD'):
        dataset = GraphSMDSegLoader(data_path, win_size, 1, mode)
    elif 'machine' in dataset:
        dataset = GraphServiceSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'MSL'):
        dataset = GraphMSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = GraphSMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SWaT'):
        dataset = GraphSWaTSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'WADI'):
        dataset = GraphWADISegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = GraphPSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'NIPS_TS_Water'):
        dataset = GraphNIPS_TS_WaterSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'NIPS_TS_Swan'):
        dataset = GraphNIPS_TS_SwanSegLoader(data_path, win_size, 1, mode)


    shuffle = mode == 'train'
    
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    return data_loader
