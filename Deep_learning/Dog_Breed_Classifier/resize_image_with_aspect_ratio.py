from PIL import Image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_PATH = 'data'
TRA

class_list = os.listdir(DATA_PATH)
size = 256, 256
for directory in class_list:
    images = os.listdir(os.path.join(DATA_PATH, directory))
    print (directory, images)
    for image in images:
        image_file = os.path.join(DATA_PATH, directory, image)
        im = Image.open(image_file)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(image_file, "JPEG")

