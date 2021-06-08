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

#####------ CLEAN DATA -----#####
# remove unused columns
df_clean = df.drop(['id','Unnamed: 32'], axis=1)

# change M into 1 and B into 0 for easy computation
df_clean.diagnosis = [1 if each == "M" else 0 for each in df_clean.diagnosis]

# divide x and y data
y = df_clean.diagnosis.values
x_data = df_clean.drop(['diagnosis'], axis = 1)

# normalize x
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values

#####------ TRAIN AND TEST DATA -----#####

train_size = int(0.7*len(df_clean)) #train & test data ratio = 70:30

x_set = x.sample(frac=1)

train_set = x_set[:train_size]
test_set = x_set[train_size:]

x_data_train, x_data_test = x.iloc[train_set.index], x.iloc[test_set.index]
y_data_train, y_data_test = y[train_set.index], y[test_set.index]

x_data_train = x_data_train.T
x_data_test = x_data_test.T
y_data_train = y_data_train.T
y_data_test = y_data_test.T


#####------ LOGISTIC REGRESSION -----#####
# initialize weight and bias
def init_weight_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.01
    return w,b

# activation function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# setting up model
def forward_backward_propagation(w,b,x_data_train,y_data_train,learning_rate,num_of_iteration):
    
    #forward backward progapagation + updating weight and bias
    for i in range(num_of_iteration):
        z = np.dot(w.T,x_data_train) + b
        y_prediction = sigmoid(z)
        
        gradient_w = np.dot(x_data_train, (y_prediction - y_data_train).T) / x_data_train.shape[1]
        gradient_b = np.mean(y_prediction - y_data_train)

        w = w - learning_rate * gradient_w
        b = b - learning_rate * gradient_b
    
    weight = w
    bias = b
    
    return weight,bias

# function for prediction
def predict(w,b,x_data_test):
    z = np.dot(w.T,x_data_test) + b
    y_prediction = sigmoid(z)
    y_prediction_new = np.zeros((1,x_data_test.shape[1]))
    
    for i in range(y_prediction.shape[1]):
        if y_prediction[0, i]<= 0.5:
            y_prediction_new[0, i] = 0
        else:
            y_prediction_new[0, i] = 1
  
    return y_prediction_new

# main function
def logistic_regression(x_data_train,y_data_train,x_data_test,y_data_test,learning_rate,num_of_iteration):
    dimension = x_data_train.shape[0]
    [w,b] = init_weight_bias(dimension)
    
    [weight,bias] = forward_backward_propagation(w,b,x_data_train,y_data_train,learning_rate, num_of_iteration)
    
    y_prediction_train = predict(weight,bias,x_data_train)
    y_prediction_test = predict(weight,bias,x_data_test)
    
    #change float data into int
    y_prediction_test = np.array(y_prediction_test,int)
    y_prediction_train = np.array(y_prediction_train,int)
    
    print("Weight Shape:",weight.shape)
    print("Bias:",bias)

    print("Train Accuracy: {} %".format(100 - np.mean(np.abs(
        y_prediction_train - y_data_train)) * 100))
    print("Test Accuracy: {} %".format(100 - np.mean(np.abs(
        y_prediction_test - y_data_test)) * 100))
    
    return weight,bias,y_prediction_test,y_prediction_train


#####------ RUNNING THE ALGORITHM -----#####
# learning_rate = 1.5; num_of_iteration = 100
[weight,bias,y_prediction_test,y_prediction_train] = logistic_regression(
    x_data_train,y_data_train,x_data_test,y_data_test,1.5,100)


#####------ IMPLEMENT TO EQUATION -----#####
def result(x):
    P = np.exp(bias + np.dot(weight.T,x))/(1 + np.exp(bias + np.dot(weight.T,x)))
    if P >= 0.5:
        y = 'M'
    elif P < 0.5:
        y = 'B'
    return P,y

# testing using random instance of dataframe
import random

# generate random number
i = random.randint(0,len(df_clean)-1)

P,y = result(x.iloc[i].values)

print('\n----- Test Result from Breast Cancer Data -----\n')
print('Instance number', i)
print('Result of Logistic Regression =',y)
print('Result on Dataframe =',df['diagnosis'].iloc[i])
if y == df['diagnosis'].iloc[i]:
    print('SAME!')
else :
    print('DIFFERENT!')
    
    
#####------ CALCULATION ERROR -----#####
error = np.zeros(len(df_clean))
P = np.zeros(len(df_clean))
for i in range(len(df_clean)-1):
    P[i],y = result(x.iloc[i].values)
    error[i] = np.absolute(df_clean["diagnosis"].iloc[i] - np.round(P[i]))

# plot error
ax = sns.countplot(error)
ax.set(title='Number of Correct and False Prediction', xticklabels = ['Correct', 'False'])

error_persentage = error.sum()/len(df_clean)*100

print('\nError Persentage:', error_persentage,'%')