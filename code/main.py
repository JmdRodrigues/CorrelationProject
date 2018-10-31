import pandas as pd
import openpyxl as xl
from pca_analysis import functionPCA
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from plot_clusters import plotClusters


colors =mcolors.CSS4_COLORS.values()

#load entire database
data = pd.read_excel('Base Dados_final.xlsx')
print(data['URQ'])
#get data from 12 months symptoms
m12 = [data[key] for key in data.keys() if('12m' in key)]
m12_keys = [key for key in data.keys() if('12m' in key)]
m7 = [data[key] for key in data.keys() if('Intensidade' in key)]
m7_keys = [key for key in data.keys() if('Intensidade' in key)]

#get data from biomechanical risk factors
bmk_r_factors_keys = ['Score', 'Carga_trabalho', 'Tempo_ciclo', 'Score_Posture']
bmk_r_factors = [data[key]/max(data[key]) for key in bmk_r_factors_keys]

All_data = m7+bmk_r_factors

All_keys = m7_keys

data2 = pd.DataFrame.from_items(zip(All_keys, All_data))
#remove all NAN values from the 12 months
data2 = data2.dropna()

print(len(data2))

data2 = data2.values


X, y_pred, X_pca, params = functionPCA(data2)

plotClusters(y_pred, X_pca, 3, data, list(colors))









