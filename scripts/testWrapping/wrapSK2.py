
from sklearn.linear_model import LinearRegression as LR
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np


# extend SK learn regression model
class SLR(LR):
	pass


height=[[4.0],[4.5],[5.0],[5.2],[5.4],[5.8],[6.1],[6.2],[6.4],[6.8]]
weight=[  42 ,  44 , 49, 55  , 53  , 58   , 60  , 64  ,  66 ,  69]

import numpy as np

X = np.asarray(height)
y = np.asarray(weight)


m1 = SLR(fit_intercept=True)
m1.fit(X,y)
print(m1.score(X,y))

def f1():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=1, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='linear'))
	# Compile model
	model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	return model

# extend SK learn regression model
class CLR(LR,KerasRegressor):
	def __init__(self,build_fn=None,**kw):
		super(CLR, self).__init__(**kw)
        
	pass
m2 = CLR(build_fn=f1,fit_intercept=False,normalize=True)
print(m2.__class__)
m2.fit(X,y)
print(m2.score(X,y))


#model = KerasRegressor(build_fn=f1, epochs=10, batch_size=10, verbose=0)