import mysql.connector
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import csv
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
from cv2 import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import ctypes
from datetime import datetime
import RPi.GPIO as GPIO


exId = 0


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_colors(image, number_of_colors, show_chart):

    modified_image = cv2.resize(
        image, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(
        modified_image.shape[0]*modified_image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize=(8, 6))
        plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
        plt.show()
    return rgb_colors


def executeId():
    global exId
    mydb = mysql.connector.connect(
        host="10.101.118.235",
        port=3306,
        user="Arm",
        password="Army27702!",
        database="armarm"
    )

    mycursor = mydb.cursor()

    chopperdb = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="shopper2542",
        database="plasmax"
    )

    choppercursor = chopperdb.cursor()

    choppercursor.execute(
        "SELECT * FROM plasmax.plasmainfo;")
    chopperResult = choppercursor.fetchall()
    print("Checking for new image ... ")

    # Query data from server
    mycursor.execute(
        "SELECT * FROM blood WHERE id=(SELECT max(id) FROM blood)")
    myresult = mycursor.fetchall()
    mycursor.execute("SELECT max(id) FROM blood")
    idd = mycursor.fetchall()
    id = int(idd[0][0])
    print(id)
    if exId != id:
        print('Detected New Image')
        exId = id
        print(myresult)
        # convert list to array
        my_array = np.array(myresult)

        # decode all image
        dbR, dbC = my_array.shape
        for i in range(dbR):
            im = Image.open(BytesIO(base64.b64decode(my_array[i, 2])))
            nameImg = my_array[i, 1] + ".png"
            im.save(
                'C:\\Users\\USER\\Desktop\\SeniorProject\\images\\' + nameImg, 'PNG')

        IMAGE_DIRECTORY = 'images'
        COLORS = {
            'ACCEPTABLE': [125, 96, 16]
        }
        images = []
        for file in os.listdir(IMAGE_DIRECTORY):
            if not file.startswith('.'):
                images.append(get_image(os.path.join(IMAGE_DIRECTORY, file)))

        def match_image_by_color(image, color, threshold, number_of_colors):
            image_colors = get_colors(image, number_of_colors, False)
            selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

            select_image = False
            for i in range(number_of_colors):
                curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
                diff = deltaE_cie76(selected_color, curr_color)
                if (diff < threshold):
                    select_image = True

            return select_image

        def show_selected_images(images, color, threshold, colors_to_match):
            #index = 1

            result = "Unacceptable"

            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            # GPIO.setup(21, GPIO.OUT) <<<---------------------- set ขาของ output

            for i in range(len(images)):
                selected = match_image_by_color(images[i],
                                                color,
                                                threshold,
                                                colors_to_match)
                if (selected):
                    # Edit code to return 0, 1 result for machine here
                    #plt.subplot(1, 5, index)
                    # plt.imshow(images[i])
                    # plt.show()
                    #index += 1
                    result = "Acceptable"
                    # GPIO.output(21, GPIO.HIGH) <<<---------------- Sending high voltage to Arduino

            return result
            # GPIO.output(21, GPIO.LOW) <<<---------------------- Sending low voltage to Arduino

        # Variable 'selected_color' can be any of COLORS['GREEN'], COLORS['BLUE'] or COLORS['YELLOW']
        plt.figure(figsize=(20, 10))
        result = show_selected_images(images, COLORS['ACCEPTABLE'], 14, 8)
        # plt.show()
        print("*************")
        print(result)
        print("*************")
        choppercursor.execute("INSERT INTO `plasmax`.`plasmainfo`(`refId`,`status`,`timeStamp`) VALUES(%s,'%s','%s');" % (
            id, result, datetime.now()))
        chopperdb.commit()
        os.remove("C:\\Users\\USER\\Desktop\\SeniorProject\\images\\" + nameImg)

    else:
        # If there's a new image in the database .
        print('Image has been read into the system .')


while True:
    executeId()
