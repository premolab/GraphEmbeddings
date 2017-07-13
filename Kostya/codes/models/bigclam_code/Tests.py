
# coding: utf-8

# Технический ноутбук для отладки и проверки того, что все корректно работает.

# # Conductance test

# In[1]:

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .Experiments import draw_matrix
from algorithms.big_clam import GetNeighborhoodConductance

get_ipython().magic('matplotlib inline')


# In[4]:

A = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]])
G = nx.Graph(A)
plt.subplot(1,2,1)
nx.draw(G,pos=nx.layout.fruchterman_reingold_layout(G))
plt.subplot(1,2,2)
draw_matrix(A)


# In[6]:

A = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]])
GetNeighborhoodConductance(A, minDeg = 0)


# 
