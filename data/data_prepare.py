# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 08:16:30 2024

@author: Xue
"""

import numpy as np
from scipy.sparse import coo_matrix,csc_matrix

# 用于将数据集中的速度npz文件转换为csv
data = np.load('./data/pems08/PEMS08.npz')
data.files
y = data['data']
# y1 = y.squeeze(axis = 2)
y1 = y[:,:,0] #flow，speed，occupancy
np.savetxt('./data/pems08/vel.csv',y1,delimiter = ',')





