import numpy as np
import os, glob, shutil
import string
import cv2
from PIL import Image, ImageFilter

raw_image_path = '/Volumes/GoogleDrive/My Drive/課業/三上/ML/Data'

for dirName in string.ascii_letters:
    files = []
    files.extend(glob.glob('{}/{}/*.png'.format(raw_image_path, dirName)))
    files.extend(glob.glob('{}/{}/*.jpg'.format(raw_image_path, dirName)))
    print('{}: {}'.format(dirName, len(files)))

    # Create directories.
    if dirName.islower():
        folderPath = 'data/preprocessed/{}-l'.format(dirName)
    else:
        folderPath = 'data/preprocessed/{}-u'.format(dirName)
    shutil.rmtree(folderPath)
    os.mkdir(folderPath)


    for i in range(len(files)):
        # Grayscale
        im = Image.fromarray(cv2.imread(files[i]))
        # print(np.asarray(im).shape)
        im = im.resize((36,36), Image.ANTIALIAS).convert('L')

        # Binarization
        std = np.asarray(im).std()
        m = np.asarray(im).mean()
        im = im.point(lambda x: 0 if x<m-1.7*std else 255, 'L')

        # Blur
        im = im.filter(ImageFilter.GaussianBlur(radius=2))

        # Save images
        im.save('{}/{}.png'.format(folderPath,i))




