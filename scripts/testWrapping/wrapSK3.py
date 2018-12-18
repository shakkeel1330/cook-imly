


class Left(object):
	def __init__(self,a='',b='',**kw):
		self.kw=[a,b]
	def say(self):
		print('** in Left params **')
		#print(self.kw)
	def get_params(self):
		return(self.kw)


# extend SK learn regression model
class Right(object):
	def __init__(self,c='',d='',**kw):
		self.kw = [c,d]
	def say(self):
		print('in Right')
		#print(self.kw)

# extend SK learn regression model
class CLR(Left,Right):
	def say(self):
		Right.say(self)



x=CLR(a=1)
print(x.say())