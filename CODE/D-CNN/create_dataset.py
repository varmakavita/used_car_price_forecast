from config import car_config as config
import shutil
import os
from imutils import paths
import random
import cv2
import csv


data_dir = config.CLEANED_DATA_DIR
output_dir = config.IMAGES_PATH
output_file = config.LABELS_PATH
threshold = config.DATASET_IMAGES_THRESHOLD
num_of_images = config.NUM_IMAGES_PER_CLASS

data = []
if os.path.exists(output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if os.path.exists(output_file):
  os.remove(output_file)

num_of_classes = 0

for file in os.listdir(data_dir):
    d = os.path.join(data_dir, file)
    class_name = d.split("/")[-1]
    if os.path.isdir(d):
        dir_images = list(paths.list_images(d))
        random.shuffle(dir_images)
        if len(dir_images) >= threshold:
            num_of_classes += 1
            for path in dir_images[0:num_of_images]:
                img = cv2.imread(path)
                img = img[:, 19:133, :]
                img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
                img_name = output_dir + "/" + path.split("/")[-1]
                year = class_name.split("_", 2)[0]
                make = class_name.split("_", 2)[1]
                model = class_name.split("_", 2)[2]
                data.append([img_name, make, model, year])

                cv2.imwrite(img_name, img)

print("[INFO] Total number of car classes : {}".format(num_of_classes))
with open(output_file, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(["Image Filename", "Make", "Model", "Year"])

    writer.writerows(data)


        