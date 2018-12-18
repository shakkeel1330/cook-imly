# https://stackoverflow.com/questions/7273568/pick-a-subclass-based-on-a-parameter

# more elegeant
# http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Factory.html

# https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python

class Dogs(object):

    @staticmethod
    def get_animal_which_makes_noise(noise='woof'):
        if noise == 'meow':
            return Cat()
        elif noise == 'woof':
            return Dog()

class Cat:
	def bark(self):
		print("2.pur")
	def wag(self):
		print("2.pur pur")


class Dog:
    def bark(self):
        print("1.WOOF")
    def wag(self):
        print("1.WOOF WOOF")
    

boby = Dog()
boby.bark() # WOOF
dody = Dog()

from types import MethodType
def barkLoud(self):
    print("Loud WOOF")
dody.bark = MethodType(barkLoud, dody)

# let us a class decorator
d = Dog()
d.wag()
dody.bark()

c = Cat()
c.wag()

print('testing dispatching')

a = Dogs.get_animal_which_makes_noise()
a.wag()

b = Dogs.get_animal_which_makes_noise('woof')
b.wag()

# call different classes via function defintion

def get_animal_which_makes_noise(noise='woof'):
        if noise == 'meow':
            return Cat()
        elif noise == 'woof':
            return Dog()

c = get_animal_which_makes_noise('meow')
c.wag()
c.bark()