from config import car_config as config
import shutil
import os
from imutils import paths
import cv2


dataset_path = config.RAW_DATA_DIR
output_path = config.CLEANED_DATA_DIR

if os.path.exists(output_path):
    shutil.rmtree(output_path, ignore_errors=True)

if not os.path.exists(output_path):
    os.mkdir(output_path)


im_paths = list(paths.list_images(dataset_path))

print(len(im_paths))
NoneType = type(None)

for i, p in enumerate(im_paths):
    im_name = p.split("/")[-1]
    new_name = im_name.split("-pic")[0]
    
    if "152x114" in im_name:
        if new_name[0:3] != "pic":
            img = cv2.imread(p)
            if type(img) != NoneType:
                h, w, c = img.shape
                if h == 114 and w == 152:
                    if not os.path.exists(output_path + "/" + new_name):
                        os.mkdir(output_path + "/" + new_name)
                    shutil.copy(p, output_path + "/" + new_name + "/" + str(i) + ".jpg")