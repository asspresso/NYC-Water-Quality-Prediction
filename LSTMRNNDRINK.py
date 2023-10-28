import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
import missingno as msno
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#导入监测数据
# df2=pd.read_csv(r'D:\pyt\数据\drink\Drinking_Water_Quality_Distribution_Monitoring_Data.csv')
# #把日期变成日期格式
# df2['Sample Date']=pd.to_datetime(df2['Sample Date'])
# df2=df2.sort_values(by='Sample Date')
# dff=df2[['Sample Date', 'Sample Site','Residual Free Chlorine (mg/L)', 'Turbidity (NTU)','Fluoride (mg/L)']]
# #重命名列名
# dff.columns=['date','Site','RFC','TUR','FLU']
# print(dff)
# #长格式数据转化为宽格式数据
# dff1=pd.pivot_table(dff,index='date',columns='Site')
# #把多层列列名转换为一层列名
# dff1.columns=['_'.join(col) for col in dff1.columns.values]
# #数据展示（白色为缺失数据）
# msno.matrix(dff1,labels=True)
# #对缺失数据进行插补
# dff1=dff1.interpolate()
# #查看数据列名和数据类型
# print(dff1.head(),dff1.columns,dff1.info())
# #对插补后数据进行展示（除去最初的几个数据外其他数据都已插补完毕）
# msno.matrix(dff1,labels=True)
# plt.show()
# #插补后数据保存
# #dff1.to_excel(r'D:\pyt\数据\DrinkingData.xlsx')

#导入训练数据
src_canada = pd.read_csv(r'D:\pyt\数据\drink\DrinkingData1122334455.csv',index_col='date',parse_dates=True)
#针对每列前面缺失的数据，用后插法补齐（最数据的缺失会引起梯度消失，模型无法训练）
src_canada=src_canada.fillna(method='bfill')
#查看数据的前二十列和列名
print(src_canada.head(20),src_canada.columns)
data = src_canada
#对数据进行标准化处理，都缩放到0-1的区间
vmean = data.apply(lambda x:np.mean(x))
vstd = data.apply(lambda x:np.std(x))
t0 = data.apply(lambda x:(x-np.mean(x))/np.std(x)).values
#LSTM
#SEQLEN表示使用前期数据的长度，dim_in表示输入数据的维度，dim_out表示输出数据的维度，pre_len表示预测的长度
SEQLEN = 14
dim_in = 6726
dim_out=6724
pre_len=30
#划分训练数据和测试数据集
X_train = np.zeros((t0.shape[0]-SEQLEN-pre_len, SEQLEN, dim_in))
Y_train = np.zeros((t0.shape[0]-SEQLEN-pre_len, dim_out),)
X_test = np.zeros((pre_len, SEQLEN, dim_in))
Y_test = np.zeros((pre_len, dim_out),)
#对基础数据进行重构
for i in range(SEQLEN, t0.shape[0]-pre_len):
    Y_train[i - SEQLEN] = t0[i][0:6724]
    X_train[i-SEQLEN] = t0[(i-SEQLEN):i]
for i in range(t0.shape[0]-pre_len,t0.shape[0]):
    Y_test[i - t0.shape[0] + pre_len] = t0[i][0:6724]
    X_test[i-t0.shape[0]+pre_len] = t0[(i-SEQLEN):i]

#搭建模型，使用256个神经元，激活函数用relu，损失函数用mean_squared_error（均方误差），模型优化算法用rmsprop，模型训练轮次epochs:500次，每次处理batch_size:30列数据
model = Sequential()
model.add(LSTM(256, input_shape=(SEQLEN, dim_in),activation='relu',recurrent_dropout=0.01))
model.add(Dense(dim_out,activation='linear'))
model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop',metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=1, batch_size=30, validation_split=0)

#查看模型训练的损失函数
history_dict=history.history
loss=history_dict['loss']
#acc=history_dict['acc']
epochs=range(1,len(loss)+1)
print(history_dict.keys())
plt.plot(epochs,loss,'b')
#plt.plot(epochs,acc,'r')
plt.axhline(y=0.8, c='blue', ls='--')
plt.axhline(y=0, c='green', ls='--')
plt.grid()
plt.show()

predates =src_canada.index[-30:].strftime('%y-%m-%d').values
#对测试数据进行预测，并还原
preddf=model.predict(X_test)*vstd[0:6724].values+vmean[0:6724].values
preddf=pd.DataFrame(preddf,columns=src_canada.columns[0:6724])
preddf.index=data.index[-30:]
print('LSTM预测结果：',preddf)

#RNN模型
import numpy as np
import pandas as pd
from keras.layers import SimpleRNN, Dense
from keras.models import Sequential
import matplotlib
import matplotlib.pyplot as plt

SEQLEN = 14
dim_in = 6726
dim_out=6724
pre_len=30
X_train = np.zeros((t0.shape[0]-SEQLEN-pre_len, SEQLEN, dim_in))
Y_train = np.zeros((t0.shape[0]-SEQLEN-pre_len, dim_out),)
X_test = np.zeros((pre_len, SEQLEN, dim_in))
Y_test = np.zeros((pre_len, dim_out),)
for i in range(SEQLEN, t0.shape[0]-pre_len):
    Y_train[i - SEQLEN] = t0[i][0:6724]
    X_train[i-SEQLEN] = t0[(i-SEQLEN):i]
for i in range(t0.shape[0]-pre_len,t0.shape[0]):
    Y_test[i - t0.shape[0] + pre_len] = t0[i][0:6724]
    X_test[i-t0.shape[0]+pre_len] = t0[(i-SEQLEN):i]


model = Sequential()
model.add(SimpleRNN(128, input_shape=(SEQLEN, dim_in),activation='tanh',recurrent_dropout=0.01))
model.add(Dense(dim_out,activation='linear'))
model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop',metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=1, batch_size=7, validation_split=0)
history_dict=history.history
loss=history_dict['loss']
#acc=history_dict['acc']
epochs=range(1,len(loss)+1)
print(history_dict.keys())
plt.plot(epochs,loss,'b')
#plt.plot(epochs,acc,'r')
plt.axhline(y=0.8, c='blue', ls='--')
plt.axhline(y=0, c='green', ls='--')
plt.grid()
plt.show()
predates =src_canada.index[-30:].strftime('%y-%m-%d').values
#对测试数据进行预测，并还原
preddf=model.predict(X_test)*vstd[0:6724].values+vmean[0:6724].values
preddf=pd.DataFrame(preddf,columns=src_canada.columns[0:6724])
preddf.index=data.index[-30:]

print('RNN预测结果：',preddf)