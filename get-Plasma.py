# Send image to server.
import mysql.connector
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import csv

mydb = mysql.connector.connect(
    host="10.101.118.235",
    port=3306,
    user="Arm",
    password="Army27702!",
    database="armarm"
)

mycursor = mydb.cursor()

# Query data from server
mycursor.execute("SELECT * FROM blood WHERE id=(SELECT max(id) FROM blood)")
myresult = mycursor.fetchall()
mycursor.execute("SELECT max(id) FROM blood")
idd = mycursor.fetchall()
id = idd[0][0]

print(id)
print(myresult)


# convert list to array
my_array = np.array(myresult)

# decode all image
dbR, dbC = my_array.shape
for i in range(dbR):
    im = Image.open(BytesIO(base64.b64decode(my_array[i, 2])))
    nameImg = my_array[i, 1] + ".png"
    im.save(nameImg, 'PNG')
