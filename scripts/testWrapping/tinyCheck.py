from tinydb import TinyDB, Query
db = TinyDB('db.json')
db.insert({'name':'soma','age':41})

db.all()

for item in db:
	print(item)


table = db.table('LinearRegression')
table.inser({'name':'jason'})