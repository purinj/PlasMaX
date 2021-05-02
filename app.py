from flask import Flask, render_template, url_for, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import base64
import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
from cv2 import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import ctypes

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///test.db"
db = SQLAlchemy(app)


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image(image):
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


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/api/imageProcess', methods=['POST'])
def imageProcess():
    if request.method == 'POST':
        Dropimage = request.form.get('image')
        img = data_uri_to_cv2_img(Dropimage)
        # ตั้งแต่ตรงนี้

        # cv2.imshow('custom window name', img)
        # cv2.waitKey(0)
        IMAGE_DIRECTORY = 'images'
        COLORS = {
            'ACCEPTABLE': [168, 105, 0]
        }
        images = []
        images.append(get_image(img))

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
            return result

        # Variable 'selected_color' can be any of COLORS['GREEN'], COLORS['BLUE'] or COLORS['YELLOW']
        plt.figure(figsize=(20, 10))
        result = show_selected_images(images, COLORS['ACCEPTABLE'], 25, 8)
        # plt.show()
        print("*************")
        print(result)
        print("*************")
    return result


if __name__ == "__main__":
    app.run(debug=True)
