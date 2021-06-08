# -*- coding: utf-8 -*-
"""
Kelompok Speedrun!

Data : Breast Cancer
Metode : Logistic Regression

Nama Anggota :
    - Abdillah Akmal Firdaus  (19/440884/TK/48678)
    - Bimo Aji Fajrianto      (19/446771/TK/49876)
    - Dhias Muhammad Naufal   (19/446774/TK/49879)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read data
df = pd.read_csv('data.csv')
print('Shape of Dataframe =',df.shape)
print(df.head())


#####------ ANALYZE DATA -----#####

df.info()

sns.countplot(x="diagnosis", data=df).set(title = 'Number of Individuals Diagnosed to Malignant or Benign')

# find corelation between each part
row = df.columns
row_mean = row[list(range(1,12))]
row_se = np.append(row[list(range(12,22))],'diagnosis')
row_worst = np.append(row[list(range(22,32))],'diagnosis')

# analyze mean data
mean_data=df[row_mean]

g = sns.pairplot(mean_data, hue='diagnosis')
g.map_diag(sns.distplot)
g.map_offdiag(plt.scatter)
g.add_legend()
g.fig.suptitle('Scatter Plot of Mean Data', fontsize = 20)
g.fig.subplots_adjust(top= 0.95);

mean_heat = sns.heatmap(mean_data.corr()) #heatmap of mean data correlation

# analyze standard error data
se_data=df[row_se]

h = sns.pairplot(se_data, hue='diagnosis')
h.map_diag(sns.distplot)
h.map_offdiag(plt.scatter)
h.add_legend()
h.fig.suptitle('Scatter Plot of Standard Error Data', fontsize = 20)
h.fig.subplots_adjust(top= 0.95);

se_heat = sns.heatmap(se_data.corr()) #heatmap of standard error data correlation

# analyze worst data
worst_data=df[row_worst]

k = sns.pairplot(worst_data, hue='diagnosis')
k.map_diag(sns.distplot)
k.map_offdiag(plt.scatter)
k.add_legend()
k.fig.suptitle('Scatter Plot of Worst Data', fontsize = 20)
k.fig.subplots_adjust(top= 0.95);

worst_heat = sns.heatmap(worst_data.corr()) #heatmap of worst data correlation


#####------ CLEAN DATA -----#####

# find null data
df.isnull()

# remove unused columns
df_clean = df.drop(['id','Unnamed: 32'], axis=1)

# change M into 1 and B into 0 for easy computation
df_clean.diagnosis = [1 if each == "M" else 0 for each in df_clean.diagnosis]
print(df_clean())

# divide x and y data
y = df_clean.diagnosis.values
x_data = df_clean.drop(['diagnosis'], axis = 1)

# normalize x
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values