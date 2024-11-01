import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


class AnalysisHierarchyProcess:
    def __init__(self, score_data, key: list, temp: np.ndarray):
        self.score_data = score_data
        self.key = key
        self.temp = temp
        self.importance = self.cal_importance()

    def cal_importance(self) -> np.ndarray:
        temp1 = 1 / self.temp
        temp2 = temp1[::-1].round(2)
        temp3 = temp2.tolist() + [1] + self.temp.tolist()
        value = np.array(temp3).reshape(-1, 1)
        return value
    
    def create_judgment_matrix(self):
        score_data = self.score_data
        k2diff = dict(zip(self.key, self.importance.flatten()))
        num_score = score_data.shape[0]
        JudementMatrix = np.array([[k2diff[score_data[j] - score_data[i]] for j in range(num_score)] for i in range(num_score)])
        return JudementMatrix
    
    def calEigenVector(self):
        J = self.create_judgment_matrix()
        eval_, evect = np.linalg.eig(J)
        lamda = np.max(eval_.real)
        ind = 0
        for i in range(len(J)):
            if lamda == eval_[i]:
                ind = i
                break
        max_vect = np.round(evect[:, ind], 2).real
        unit_vect = np.round(max_vect / (np.sum(max_vect)), 4).real
        return unit_vect


class MSET:
    def __init__(self, history_data: np.array, score_data:np.array, 
         key:list =[-6, -4, -2, 0, 2, 4, 6], temp:np.array= np.array([3, 5, 7])):
        MemoryMatrix = self.create_memory_matrix(history_data)
        self.D, self.factor = self.calFactor(MemoryMatrix)
        self.p_min, self.p_max = np.min(MemoryMatrix, axis=0), np.max(MemoryMatrix, axis=0)
        self.p_ptp = self.p_max - self.p_min
        self.AHP = AnalysisHierarchyProcess(score_data, key, temp)
        self.W = self.AHP.calEigenVector()
    
    def create_memory_matrix(self, history_data):
        sample_n, ratio, n_clusters = 4, 0.1, 3
        
        def sample_data(history_data):
            data = history_data
            minmax = MinMaxScaler()
            X = minmax.fit_transform(data)
            sample_arr = []
            m = np.arange(1, self.sample_n + 1) * (1 / self.sample_n)
            for i in range(X.shape[1]):
                for j in range(1, sample_n + 1):
                    np_bool = np.abs(X[:, i] - m[j - 1]) < ratio
                    if np_bool.any():
                        sample_arr.append(np.argmax((np_bool)))
            sample_arr_unique = np.unique(np.array(sample_arr))
            data_interval = history_data[sample_arr_unique]
            return data_interval


        def km_data(data_interval):
            km = KMeans(n_clusters=n_clusters, random_state=9).fit(data_interval)    
            centroid = km.cluster_centers_   # 质心
            labels = km.labels_  # 聚类标签
            ue, fi = np.unique(labels, return_index=True)
            return data_interval[fi]
        
        data_interval = sample_data(history_data)
        MemoryMatrix = km_data(data_interval)
        return MemoryMatrix

    def keni(self, x):
        return np.asmatrix(x).I

    def calFactor(self, M: np.array):
        data = M.copy()
        data = data.round(3)
        mmr = MinMaxScaler()
        X = mmr.fit_transform(data)
        DT = X.T
        D = X
    
        factor = DT * self.keni(distance.cdist(D, D, "euclidean"))
        return D, factor
    
    def predict(self, new_data: np.array):
        new_data = new_data.astype(np.float32)
        new_data = new_data.reshape(1, -1)
        data_no_constrain = new_data.reshape(1, -1)
        tr = new_data.reshape(1, -1)
        
        new_data = np.where(new_data > self.p_max, self.p_max, new_data)
        new_data = np.where(new_data < self.p_min, self.p_min, new_data)
        
        new_data = (new_data - self.p_min) / self.p_ptp
        data_no_constrain = (data_no_constrain - self.p_min) / self.p_ptp
        
        predict_value = np.asmatrix(self.factor) * distance.cdist(self.D, new_data, 'euclidean')
        pr = np.multiply(np.array(predict_value).reshape(1, -1), self.p_ptp) + self.p_min
    
        # similaruty
        similaruty = 1 / (1 + abs(data_no_constrain.flatten() - predict_value.T))
        similaruty_ = np.array(similaruty)[0]
        
        # residual
        residual = (tr - pr)[0]
        
        # healthy
        healthy = 1 / (1 + np.abs(np.sqrt(np.sum(W * (residual / p_ptp) ** 2))))
        return tr[0].tolist(), pr[0].tolist(), similaruty_.tolist(), residual.tolist(), healthy
