import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import pickle
import warnings
import datetime
from sklearn.impute import SimpleImputer
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
data = pd.read_csv(r'D:\pyt\数据\drink\DrinkingData1122334455.csv',index_col='date',parse_dates=True)
print(data.shape)
data=data.astype(float)
#针对每列前面缺失的数据，用后插法补齐（最数据的缺失会引起梯度消失，模型无法训练）
# df1=pd.read_csv(r'D:\pyt\数据\df_prcp.csv')
# df2=pd.read_csv(r'D:\pyt\数据\df_tavg.csv')
# df=pd.merge(df1,df2,left_on='date',right_on='date')
# df=df[['date','prcp','tavg']]
# df['date']=pd.to_datetime(df['date'])
# print(df.info(),src_canada.shape)
# data=pd.merge(src_canada,df,left_on=src_canada.index,right_on='date')
# data.index=data['date']
# # df=data[data.apply(lambda x:(abs(x-np.mean(x))/np.std(x))<=3,axis=0)]
# df=data.drop(columns='date')
# print(df.index,df.columns,df.shape)
#df=df.iloc[:-30]
#imp=SimpleImputer(strategy='mean')
#data=imp.fit(data)
#print(data.shape[0],data.shape[1])
parts=14
this_one=data.iloc[parts:]
#print(this_one.shape)
bak_index=this_one.index
#print(this_one.index.shape)
for k in range(1,parts+1):
    last_one=data.iloc[(parts-k):(this_one.shape[0]-k+parts)]
    #print(last_one.index)
    this_one.set_index(last_one.index,drop=True,inplace=True)
    this_one=this_one.join(last_one,lsuffix='',rsuffix="_p" + str(k))
    print(this_one.shape)
    #print(list(this_one))
this_one.set_index(bak_index, drop=True, inplace=True)
this_one = this_one.fillna(0)
t0 = this_one.iloc[:, 0:6724]

print(t0)
t0_min = t0.apply(lambda x: np.min(x), axis=0).values
t0_ptp = t0.apply(lambda x: np.ptp(x), axis=0).values
print(t0_min.shape)
print(t0_ptp.shape)
this_one = this_one.apply(lambda x: (x - np.min(x)) / np.ptp(x), axis=0)
test_data = this_one.iloc[-30:]
#print(test_data)
train_data = this_one.iloc[:-30]
#print(train_data)
train_y_df = train_data.iloc[:, 0:6724]
train_y_df=train_y_df.interpolate()
w=train_y_df.columns
for i in range(train_y_df.shape[1]):
    p=train_y_df[w[i]].isnull().sum()
    if p>0:
        print(w[i],p)
# print(train_y_df)
train_y = np.array(train_y_df)
print(train_y)
train_x_df = train_data.iloc[:, 6724:]
print(train_x_df.shape)
train_x = np.array(train_x_df)

test_y_df = test_data.iloc[:, 0:6724]
test_y = np.array(test_y_df)
test_x_df = test_data.iloc[:, 6724:]
#print(test_x_df.shape)
test_x = np.array(test_x_df)
test_y_real = t0.iloc[-30:]
#print(test_y_real)
# with open('VOC/train_x.pkl', 'wb') as f:
#     pickle.dump(train_x, f)
# with open('VOC/train_y.pkl', 'wb') as f:
#     pickle.dump(train_y, f)
# with open('VOC/test_x.pkl', 'wb') as f:
#     pickle.dump(test_x, f)
# with open('VOC/test_y_real.pkl', 'wb') as f:
#     pickle.dump(test_y_real, f)
# with open('VOC/t0_ptp.pkl', 'wb') as f:
#     pickle.dump(t0_ptp, f)
# with open('VOC/t0_min.pkl', 'wb') as f:
#     pickle.dump(t0_min, f)
# with open('VOC/train_x.pkl', 'rb') as f:
#     train_x = pickle.load(f)
# with open('VOC/train_y.pkl', 'rb') as f:
#     train_y = pickle.load(f)
# with open('VOC/history.pkl', 'rb') as f:
#     history = pickle.load(f)
init = keras.initializers.glorot_uniform(seed=1)
simple_adam = keras.optimizers.Adam()
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=512, input_dim=94166, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dense(units=256, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dense(units=128, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dense(units=64, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dense(units=32, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dense(units=16, kernel_initializer=init, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(units=6724, kernel_initializer=init, activation='tanh'))
model.compile(loss='mse', optimizer=simple_adam, metrics=['accuracy'])
history=model.fit(train_x, train_y, epochs=2, batch_size=30, shuffle=True, verbose=True,validation_data=(test_x,test_y))


# with open('VOC/model.pkl', 'wb') as f:
#     pickle.dump(model, f)
# with open('VOC/model.pkl', 'rb') as f:
#     model = pickle.load(f)
# with open('VOC/test_x.pkl', 'rb') as f:
#     test_x = pickle.load(f)
# with open('VOC/t0_min.pkl', 'rb') as f:
#     t0_min = pickle.load(f)
# with open('VOC/t0_ptp.pkl', 'rb') as f:
#     t0_ptp = pickle.load(f)
pred_y = model.predict(test_x)
pred_y = (pred_y*t0_ptp)+t0_min
# with open('VOC/pred_y4.pkl', 'wb') as f:
#     pickle.dump(pred_y, f)
# with open('VOC/history4.pkl', 'wb') as f:
#     pickle.dump(history, f)


predates =data.index[-30:].strftime('%y-%m-%d').values

preddf=pd.DataFrame(pred_y,columns=t0.columns)
preddf.index=data.index[-30:]

print('DNN预测结果：',preddf)