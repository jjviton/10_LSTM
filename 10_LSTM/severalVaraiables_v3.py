#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://youtu.be/tepxdcepTbY
"""
@author: J3viton
Code tested on Tensorflow: 2.2.0
    Keras: 2.4.3
dataset: https://finance.yahoo.com/quote/GE/history/
Also try S&P: https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC
"""


# In[2]:




import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
import seaborn as sns



# In[3]:







df32 = pd.DataFrame(columns=['instrumento', 'divergencia', 'fecha'])

# In[4]:


import sys
import os
sys.path.insert(0,"C:\\Users\\INNOVACION\\Documents\\J3\\100.- cursos\\Quant_udemy\\programas\\Projects\\libreria")
import quant_j3_lib as quant_j




# ### Instrumentos

# In[5]:


tickers5 = ['mrl.mc']

#VALORES DEL SP500
###(tickers_sp500)): 

# In[6]:


