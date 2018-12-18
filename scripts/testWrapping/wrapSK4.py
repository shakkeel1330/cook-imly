
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Lasso

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np

X=np.asarray([[4.0],[4.5],[5.0],[5.2],[5.4],[5.8],[6.1],[6.2],[6.4],[6.8]])
y = np.asarray([  42 ,  44 , 49, 55  , 53  , 58   , 60  , 64  ,  66 ,  69])


skmap = {
  'Lasso': 'f1',
  'LinearRegression': 'f1'
}

def f1():
	# create model
	model = Sequential()
	model.add(Dense(2, input_dim=1, activation='linear'))
	model.add(Dense(1, activation='linear'))
	# Compile model
	model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	return model

x1 = LR()
x2 = Lasso()

class myKSR(KerasRegressor):
	def to_json(self):
		return self.__dict__['build_fn']().to_json()


def dope(obj):
	print('class names')
	print(obj.__class__.__name__)
	print('params')
	print(obj.get_params())
	#model = KerasRegressor(build_fn=f1, epochs=10, batch_size=10, verbose=0)
	model = myKSR(build_fn=f1, epochs=10, batch_size=10, verbose=0)
	return model


model = dope(x1)

print(model.fit(X,y))
print(model.score(X,y))


#model = KerasRegressor(build_fn=f1, epochs=10, batch_size=10, verbose=0)