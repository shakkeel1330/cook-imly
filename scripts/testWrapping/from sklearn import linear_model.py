from sklearn import linear_model
from sklearn import datasets

height=[[4.0],[4.5],[5.0],[5.2],[5.4],[5.8],[6.1],[6.2],[6.4],[6.8]]
weight=[  42 ,  44 , 49, 55  , 53  , 58   , 60  , 64  ,  66 ,  69]
print("height weight")
for row in zip(height, weight):
    print(row[0][0],"->",row[1])


import numpy as np

X = np.asarray(height)
y = np.asarray(weight)

reg=linear_model.LinearRegression()
reg.fit(X,y)


m=reg.coef_[0]
b=reg.intercept_
print("slope=",m, "intercept=",b)

yh = reg.predict(X)
print('sk reg score: ',reg.score(X,y))

from sklearn.linear_model import LinearRegression
m2 = LinearRegression()


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import numpy

def myLinearRegression():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=1, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='linear'))
	# Compile model
	model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	return model

model = KerasRegressor(build_fn=myLinearRegression, epochs=10, batch_size=10, verbose=0)
import numpy as np

model.fit(X,y)
yhk = model.predict(X)
yh = reg.predict(X)

kmodel = myLinearRegression()

m2 = KerasRegressor(build_fn=None, epochs=10, batch_size=10, verbose=0)


