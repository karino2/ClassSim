import numpy as np
from scipy.misc import imresize, imread
from skimage.color import gray2rgb

SIZE = 224

def preprocess(img_arr):
    '''
    input : image as numpy array
    output : preprocessed image numpy array
    '''

    # this code is no more necessary because imread now specify mode.DataSet
    # But keep for backward compatibility (for some old notebook cell).
    if len(img_arr.shape) == 2:
        img_arr = gray2rgb(img_arr)

    height, width, chan = img_arr.shape

    centery = height // 2
    centerx = width // 2
    radius = min((centerx, centery))
    img_arr = img_arr[centery-radius:centery+radius, centerx-radius:centerx+radius]
    img_arr = imresize(img_arr, size=(SIZE, SIZE), interp='bilinear')
    #Convert to float32
    img_arr = np.array(img_arr, dtype=np.float32)
    #Rescale
    img_arr /= 255.

    return img_arr


class DataSet:
    '''
    Data preparation for prediction
    '''
    def __init__(self):
        pass

    def chunked(self, iterable, n):
        return [iterable[x:x + n] for x in range(0, len(iterable), n)]

    def load_and_preprocess(self, file_path):
        img = imread(file_path, mode="RGB")
        try:
            return preprocess(img)
        except ValueError as err:
            raise ValueError("Corrupted file: {0}, {1}".format(file_path, err))
    
    def files_to_dataset(self, file_paths):
        test_data = []
        shape = None

        for file_path in file_paths:
            img = self.load_and_preprocess(file_path)
            test_data.append(img)

        test_data = np.array(test_data).astype(np.float32)

        return test_data


## Generator
from keras.preprocessing.image import ImageDataGenerator

def create_generators():
    traingen = ImageDataGenerator(
        preprocessing_function=preprocess,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=0.2,
        rotation_range=15,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )
    validgen = ImageDataGenerator(
        preprocessing_function=preprocess
    )
    return traingen, validgen