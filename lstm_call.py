import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
x=[[1],[2],[3]]#特征
y=[2,4,6]#标签
x = np.array( x )
y_train = np.array(y )
x_train = np.reshape( x, (x.shape[0], x.shape[1], 1) )#Lstm调用库函数必须要进行维度转换
model = Sequential()
model.add( LSTM( 100, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True) )
model.add( LSTM( 20, return_sequences=False ) )
model.add( Dropout( 0.2 ) )
model.add( Dense( 1 ) )
model.add( Activation( 'linear' ) )
model.compile( loss="mse", optimizer="rmsprop" )
model.fit( x_train, y_train, epochs=200, batch_size=1)#参数依次为特征，标签，训练循环次数，小批量（一次放入训练的数据个数）
test=[[1.5]]
test=np.array(test)
test = np.reshape( test, (test.shape[0], test.shape[1], 1) )#维度转换
res = model.predict( test )
print( res )
