import os
from PIL import Image
import sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import shutil
from random import shuffle

RAW_DATA_PATH = 'data'
FINAL_DATA_PATH = 'data_split'
TRAINING_DATA_PATH = 'train'
VALIDATION_DATA_PATH = 'validation'
TEST_DATA_PATH = 'test'


def get_raw_data_stats():
    """
    This function prints out some basic stats about the raw dataset
    """
    class_list = os.listdir(RAW_DATA_PATH)
    print ('Total Class Count    :   ', len(class_list))

    if sys.version_info[0] == 2:
        min_data_cnt = sys.maxint
        min_height = sys.maxint
        min_width = sys.maxint
    else:
        min_data_cnt = sys.maxsize
        min_height = sys.maxsize
        min_width = sys.maxsize
    min_data_class = None
    max_data_cnt = 0
    max_data_class = None
    max_height = 0
    max_width = 0
    distinct_image_formats = set()
    total_image_cnt = 0

    for directory in class_list:
        images = os.listdir(os.path.join(RAW_DATA_PATH, directory))
        # print (directory, len(images))
        data_cnt = len(images)
        if data_cnt < min_data_cnt:
            min_data_cnt = data_cnt
            min_data_class = directory
        if data_cnt > max_data_cnt:
            max_data_cnt = data_cnt
            max_data_class = directory
        for image in images:
            total_image_cnt += 1
            img_format = image.split('.')[-1]  # format of current file
            distinct_image_formats.add(img_format)  # update unique img format types
            im = Image.open(os.path.join(RAW_DATA_PATH, directory, image))
            width, height = im.size
            if width < min_width:
                min_width = width
            if height < min_height:
                min_height = height
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height
            # print image, width, height
    print ('\n')
    print ('There are total {0} Images across all classes'.format(total_image_cnt))
    print ('Class \"{0}\" with {1} images has the Maximum number of data points'.format(max_data_class, max_data_cnt))
    print ('Class \"{0}\" with {1} images has the Minimum number of data points'.format(min_data_class, min_data_cnt))
    print ('\n')
    print ('Minimum Image Width    :   ', min_width)
    print ('Minimum Image Height    :   ', min_height)
    print ('Maximum Image Width  :   ', max_width)
    print ('Maximum Image Height  :   ', max_height)
    print('\n')
    print ('Distinct Image Formats in the dataset   :   ', list(distinct_image_formats))


def generate_synthetic_data(target_data_cnt):
    """
    This function generates synthetic images using data augmentation techniques
    """
    tmp_dir = 'tmp'
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    for class_name in os.listdir(RAW_DATA_PATH):
        curr_path = os.path.join(RAW_DATA_PATH, class_name)
        images = os.listdir(curr_path)  # data points ( images) in current class ( directory )
        curr_data_point_cnt = len(images)  # count of data points in the current class
        syn_img_prefix = class_name.split('.')[1] + '_' + 'synthetic'  # synthetic img prefix
        #   ## generate 1 synthetic image cyclically till the data point count < target_data_cnt
        while curr_data_point_cnt < target_data_cnt:
            #   ## keep generating 1 synthetic image for each of the images till the total data point cnt = target cnt
            for image in images:
                print ('Original Image :    ', image)
                try:
                    img = load_img(os.path.join(curr_path, image))  # PIL image
                    x = img_to_array(img)  # numpy array of shape (3, current_image_width, current_image_height)
                    x = x.reshape((1,) + x.shape)  # numpy array of shape (1, 3, curr_image_width, curr_image_height)
                except TypeError:
                    continue
                #   ##  generate batches of randomly transformed images and save it to the current class folder
                for batch in datagen.flow(x, batch_size=1, save_to_dir=tmp_dir, save_prefix=syn_img_prefix,
                                          save_format='jpg'):
                    curr_data_point_cnt += 1
                    break
                #   ## stop if total data points have reached target data points
                if curr_data_point_cnt >= target_data_cnt:
                    break
        #   ##  move the synthetic images from the tmp_dir to the actual class directory
        for f in os.listdir(tmp_dir):
            shutil.move(os.path.join(tmp_dir, f), curr_path)


def split_train_validate_test_set(train_size, validation_size, test_size):
    """
    This function splits the data into train, validation and test data in their respective data folders
    """

    #   ## create the directory structure for the split
    if not os.path.isdir(FINAL_DATA_PATH):
        os.mkdir(FINAL_DATA_PATH)
    train_path = os.path.join(FINAL_DATA_PATH, TRAINING_DATA_PATH)
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
    validation_path = os.path.join(FINAL_DATA_PATH, VALIDATION_DATA_PATH)
    if not os.path.isdir(validation_path):
        os.mkdir(validation_path)
    test_path = os.path.join(FINAL_DATA_PATH, TEST_DATA_PATH)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    #   ## iterate over the raw data and perform the split
    for directory in os.listdir(RAW_DATA_PATH):

        #   ##  create the class directory in train/validation/test folder
        if not os.path.isdir(os.path.join(train_path, directory)):
            os.mkdir(os.path.join(train_path, directory))
        if not os.path.isdir(os.path.join(validation_path, directory)):
            os.mkdir(os.path.join(validation_path, directory))
        if not os.path.isdir(os.path.join(test_path, directory)):
            os.mkdir(os.path.join(test_path, directory))

        #   ##  initialize variables before parsing the data-points in the class
        curr_path = os.path.join(RAW_DATA_PATH, directory)
        images = os.listdir(curr_path)  # list of images in the current class ( directory )
        shuffle(images)  # shuffling the list to introduce randomness in the list split
        total_data_cnt = len(images)  # count of data points in this class ( directory )

        #   ## calculating actual train/validation/test data point counts according to percentages required
        train_data_cnt = round(train_size * total_data_cnt)
        validation_data_cnt = round(validation_size * total_data_cnt)
        test_data_size = round(test_size * total_data_cnt)

        #   ## iterate over all the all the images in the class ( directory ) and perform the split
        cntr = 1  # keep track of the number of data points parsed
        for image in images:
            if cntr <= train_data_cnt:
                shutil.copyfile(os.path.join(RAW_DATA_PATH, directory, image), os.path.join(train_path, directory, image))
                cntr += 1
            elif cntr <= (train_data_cnt + validation_data_cnt):
                shutil.copyfile(os.path.join(RAW_DATA_PATH, directory, image), os.path.join(validation_path, directory, image))
                cntr += 1
            else:
                shutil.copyfile(os.path.join(RAW_DATA_PATH, directory, image), os.path.join(test_path, directory, image))
                cntr += 1


def resize_images():
    """
    This function re-sizes images maintaining the aspect ratio from the original image and saves it in the original
    location
    """
    class_list = os.listdir(RAW_DATA_PATH)
    size = 224, 224
    for directory in class_list:
        images = os.listdir(os.path.join(RAW_DATA_PATH, directory))
        print (directory, images)
        for image in images:
            image_file = os.path.join(RAW_DATA_PATH, directory, image)
            im = Image.open(image_file)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(image_file, "JPG")


get_raw_data_stats()
# generate_synthetic_data(96)
# get_raw_data_stats()
# split_train_validate_test_set(0.7, 0.2, 0.1)
