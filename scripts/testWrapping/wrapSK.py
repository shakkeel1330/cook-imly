
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Lasso

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import numpy

height=[[4.0],[4.5],[5.0],[5.2],[5.4],[5.8],[6.1],[6.2],[6.4],[6.8]]
weight=[  42 ,  44 , 49, 55  , 53  , 58   , 60  , 64  ,  66 ,  69]
print("height weight")
for row in zip(height, weight):
    print(row[0][0],"->",row[1])


import numpy as np

X = np.asarray(height)
y = np.asarray(weight)


skmap = {
  'Lasso': 'f1',
  'LinearRegression': 'f2'
}

def f1():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=1, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='linear'))
	# Compile model
	model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	return model
def f2():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=1, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='linear'))
	# Compile model
	model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	return model

x1 = LinearRegression()
x2 = Lasso()

def idly(obj):
	print('class names')
	print(obj.__class__.__name__)
	print('params')
	print(obj.get_params())
	model = KerasRegressor(build_fn=f1, epochs=10, batch_size=10, verbose=0)
	return model

def idly2(obj):
	print('class names')
	print(obj.__class__.__name__)
	print('params')
	print(obj.get_params())
	model = KerasRegressor(build_fn=f1, epochs=10, batch_size=10, verbose=0)
	obj.predict = model.predict
	obj.score = model.score
	obj.fit = model.fit
	return obj

def ify(obj):
	def wrap(*args,**kwargs):
		print('class names')
		print(obj.__class__.__name__)
		print('params')
		print(obj.get_params())
		model = KerasRegressor(build_fn=f1, epochs=10, batch_size=10, verbose=0)
		obj.predict = model.predict
		obj.score = model.score
		obj.fit = model.fit
		return obj
	return wrap


from sklearn.linear_model import LinearRegression as LR



x2 = LinearRegression()
x2 = idly2(x2)

mwrappedModel = idly(x1)
mwrappedModel.fit(X,y)
mwrappedModel.score(X,y)


#model = KerasRegressor(build_fn=f1, epochs=10, batch_size=10, verbose=0)