import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict, Any


class AnalysisHierarchyProcess:
    """层次分析法 (AHP - Analytic Hierarchy Process)
    用于根据专家打分或相对重要性标度计算各指标的权重向量。
    """
    def __init__(self, score_data: np.ndarray, key: list, temp: np.ndarray):
        self.score_data = score_data
        self.key = key
        self.temp = temp
        self.importance = self.cal_importance()

    def cal_importance(self) -> np.ndarray:
        temp1 = 1.0 / self.temp
        temp2 = temp1[::-1].round(2)
        temp3 = temp2.tolist() + [1.0] + self.temp.tolist()
        return np.array(temp3).reshape(-1, 1)
    
    def create_judgment_matrix(self) -> np.ndarray:
        score_data = self.score_data
        k2diff = dict(zip(self.key, self.importance.flatten()))
        num_score = score_data.shape[0]
        judgment_matrix = np.array(
            [[k2diff.get(score_data[j] - score_data[i], 1.0) for j in range(num_score)] for i in range(num_score)]
        )
        return judgment_matrix
    
    def cal_eigen_vector(self) -> np.ndarray:
        """计算判断矩阵的最大特征值对应的归一化特征向量（权重向量）"""
        J = self.create_judgment_matrix()
        eval_, evect = np.linalg.eig(J)
        max_idx = np.argmax(eval_.real)
        max_vect = np.abs(evect[:, max_idx].real)
        unit_vect = max_vect / np.sum(max_vect)
        return unit_vect.round(4)


class MSET:
    """多元状态估计技术 (MSET - Multivariate State Estimation Technique)
    结合 AHP 权重评估工业设备传感器时序状态与健康度。
    """
    def __init__(
        self,
        history_data: np.ndarray,
        score_data: np.ndarray,
        key: list = [-6, -4, -2, 0, 2, 4, 6],
        temp: np.ndarray = np.array([3.0, 5.0, 7.0])
    ):
        self.history_data = history_data
        self.sample_n = 4
        self.n_clusters = 3
        self.ratio = 0.1

        # 1. 构建记忆矩阵 (Memory Matrix)
        self.MemoryMatrix = self.create_memory_matrix(history_data)
        self.D, self.factor = self.cal_factor(self.MemoryMatrix)
        
        # 2. 计算极值与动态极差
        self.p_min = np.min(self.MemoryMatrix, axis=0)
        self.p_max = np.max(self.MemoryMatrix, axis=0)
        self.p_ptp = np.where((self.p_max - self.p_min) == 0, 1e-6, self.p_max - self.p_min)
        
        # 3. 计算 AHP 权重
        self.AHP = AnalysisHierarchyProcess(score_data, key, temp)
        self.W = self.AHP.cal_eigen_vector()
    
    def create_memory_matrix(self, history_data: np.ndarray) -> np.ndarray:
        """通过分位数抽样与 KMeans 聚类构建覆盖全工况的记忆矩阵"""
        data = history_data.copy()
        minmax = MinMaxScaler()
        X = minmax.fit_transform(data)
        sample_arr = []
        m = np.arange(1, self.sample_n + 1) * (1.0 / self.sample_n)
        
        for i in range(X.shape[1]):
            for j in range(self.sample_n):
                np_bool = np.abs(X[:, i] - m[j]) < self.ratio
                if np_bool.any():
                    sample_arr.append(np.argmax(np_bool))
                    
        sample_arr_unique = np.unique(np.array(sample_arr))
        if len(sample_arr_unique) < self.n_clusters:
            data_interval = history_data
        else:
            data_interval = history_data[sample_arr_unique]

        # 聚类精简记忆矩阵
        actual_clusters = min(self.n_clusters, len(data_interval))
        km = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10).fit(data_interval)
        labels = km.labels_
        _, first_indices = np.unique(labels, return_index=True)
        return data_interval[first_indices]

    def cal_factor(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算 MSET 权重因子矩阵: factor = D.T * inv(K(D, D))"""
        data = M.copy().round(3)
        mmr = MinMaxScaler()
        X = mmr.fit_transform(data)
        D = X
        DT = X.T
        
        kernel_matrix = distance.cdist(D, D, "euclidean")
        # 使用伪逆替代直接求逆，防止奇异矩阵
        inv_kernel = np.linalg.pinv(kernel_matrix)
        factor = np.dot(DT, inv_kernel)
        return D, factor
    
    def predict(self, new_data: np.ndarray) -> Dict[str, Any]:
        """对新输入传感器数据进行状态估计与健康度评估"""
        new_data = np.asarray(new_data, dtype=np.float32).reshape(1, -1)
        raw_input = new_data.copy()
        
        # 边界截断与归一化
        clipped_data = np.clip(new_data, self.p_min, self.p_max)
        norm_data = (clipped_data - self.p_min) / self.p_ptp
        
        # MSET 非线性状态估计: y_hat = factor * K(D, x)
        kernel_x = distance.cdist(self.D, norm_data, 'euclidean')
        pred_norm = np.dot(self.factor, kernel_x).reshape(1, -1)
        
        # 反归一化得到物理量预测估计值
        estimated_val = pred_norm * self.p_ptp + self.p_min
        
        # 计算相似度与残差
        similarity = 1.0 / (1.0 + np.abs(norm_data.flatten() - pred_norm.flatten()))
        residual = (raw_input - estimated_val).flatten()
        
        # 加权健康指数 (0 ~ 1 之间，越接近 1 说明设备状态越健康)
        weighted_residual = np.sum(self.W * ((residual / self.p_ptp) ** 2))
        healthy_index = 1.0 / (1.0 + np.sqrt(weighted_residual))
        
        return {
            "raw_input": raw_input.flatten().tolist(),
            "estimated_val": estimated_val.flatten().tolist(),
            "similarity": similarity.tolist(),
            "residual": residual.tolist(),
            "healthy_index": float(healthy_index),
            "weights": self.W.tolist()
        }


if __name__ == '__main__':
    # 模拟 100 条 3 通道传感器历史正常运行数据 (如温度、压力、振动)
    np.random.seed(42)
    hist_telemetry = np.random.normal(loc=[50.0, 1.2, 0.05], scale=[2.0, 0.1, 0.01], size=(100, 3))
    
    # 传感器重要性评分 (用于 AHP 权重计算)
    sensor_scores = np.array([8, 6, 9])
    
    # 初始化并训练 MSET-AHP 模型
    mset = MSET(history_data=hist_telemetry, score_data=sensor_scores)
    
    # 正常数据推理测试
    test_normal = np.array([50.5, 1.22, 0.052])
    res_normal = mset.predict(test_normal)
    print("--- 正常状态估计 ---")
    print(f"健康度指数: {res_normal['healthy_index']:.4f}")
    print(f"残差: {np.round(res_normal['residual'], 4)}")

    # 异常数据推理测试 (模拟传感器突发异常)
    test_anomaly = np.array([65.0, 1.85, 0.15])
    res_anomaly = mset.predict(test_anomaly)
    print("\n--- 异常状态估计 ---")
    print(f"健康度指数: {res_anomaly['healthy_index']:.4f}")
    print(f"残差: {np.round(res_anomaly['residual'], 4)}")
