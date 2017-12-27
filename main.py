import numpy as np
from PIL import Image

# read the file
raw_data = list(np.genfromtxt("data/semeion.data", delimiter=' ', dtype=None))
digit_data = []
digit_target = []

# data preprocess
for row in raw_data:
  row = list(row)
  digit_data.append(np.asarray(row[0:256]).reshape([16,16]))
  digit_target.append(row[256:267].index(1))
digit_data = np.asarray(digit_data)
digit_target = np.asarray(digit_target)

# save array to image file
counter = np.zeros(10,dtype=np.int)
for i in range(len(digit_target)):
  digit = digit_target[i]
  im = Image.fromarray(digit_data[i]*255)
  im.convert("RGB").save("image/{}-{}.jpg".format(digit,counter[digit]))
  counter[digit] = counter[digit] + 1
