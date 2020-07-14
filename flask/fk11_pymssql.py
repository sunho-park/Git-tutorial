import pymssql as ms

conn = ms.connect(server='127.0.0.1', user='bit2', 
                    password='0423', database = 'bitdb')


print('끗')


cursor = conn.cursor()

cursor.execute("SELECT * FROM iris2;")

row = cursor.fetchone()     #150여개중 1줄가져옴

while row :
    print("첫컬럼 : , %s, 둘컬럼 : %s" %(row[0], row[1]))
    row = cursor.fetchone()


conn.close()




