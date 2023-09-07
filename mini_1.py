import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout


data=pd.read_csv('finale-data.csv')
data.head()
data=data.dropna()
trainData=data.iloc[:,32:33].values
sc=MinMaxScaler(feature_range=(0,1))
trainData=sc.fit_transform(trainData)
trainData.shape
X_train=[]
Y_train=[]
for j in range(0,281,10):
    for i in range(3,10):
        X_train.append(trainData[i-3+j:i+j,0])
        Y_train.append(trainData[i+j,0])
X_train,Y_train=np.array(X_train),np.array(Y_train)
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_train.shape
model=Sequential()
model.add(LSTM(units=100,return_sequences=True,input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')
hist=model.fit(X_train,Y_train,epochs=50,batch_size=32,verbose=2)
plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper left')
plt.show()
testData=pd.read_csv('45.csv')
# testData['Close']=pd.to_numeric(testData.Close,errors='coerce')
testData=testData.dropna()
testData=testData.iloc[:,32:33]
y_test=testData.iloc[:,0:].values
inputClosing=testData.iloc[:,0:].values
inputClosing_scaled=sc.transform(inputClosing)
inputClosing_scaled.shape
x_test=[]
length=len(testData)
print(length)
timestamp=3
for i in range(timestamp,length+1):
    x_test.append(inputClosing_scaled[i-timestamp:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
x_test.shape
y_pred=model.predict(x_test)
predicted_price=sc.inverse_transform(y_pred)
plt.plot(y_test,color='red',label='Actual Stock Price')
plt.plot(predicted_price,color='green',label='Predicted Stock Price')
plt.title('Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()