#-*- coding=utf8 -*-
import pandas as pd
from pandas import DataFrame
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

test_set = pd.read_csv('raw/TestSet.csv')
train_set = pd.read_csv('raw/TrainingSet.csv')
test_subset = pd.read_csv('raw/TestSubset.csv')
train_subset = pd.read_csv('raw/TrainingSubset.csv')

train = train_set.drop(['EbayID','QuantitySold','SellerName'], axis=1)
train_target = train_set['QuantitySold']
test = test_set.drop(['EbayID','QuantitySold','SellerName'], axis=1)
test_target = test_set['QuantitySold']

train=np.array(train/train.max(0))
train_target=np.array(train_target)
test=np.array(test/test.max(0))
test_target=np.array(test_target)

# 获取总特征数
_, n_features = train.shape

# 定义多层感知机的网络 3个输入节点，20个隐藏节点，8个输出节点
model = Sequential()
model.add(Dense(256, input_dim=25, activation='relu'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mse', optimizer='SGD', metrics=['accuracy'])

history = model.fit(train, train_target, nb_epoch=20, batch_size=1000)

loss, accuracy = model.evaluate(train, train_target)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

pred = model.predict(test)
accuracy = np.mean(np.argmax(pred) == np.argmax(test_target))
print("Accuracy: %.2f%%" % (accuracy*100))