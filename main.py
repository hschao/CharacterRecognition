import numpy as np
from PIL import Image
from skimage import feature
from sklearn import tree,neighbors,naive_bayes
from sklearn.model_selection import train_test_split, cross_val_score

# read the file
raw_data = list(np.genfromtxt("data/optdigits.tra", delimiter=',', dtype=None))
image = []
digit_data = []
digit_label = []

# data preprocess
for row in raw_data:
    row = list(row)
    # store label
    digit_label.append(row[-1])

    # transfrom image to feature
    img = np.asarray(row[0:64]).reshape([8,8]).astype('uint8')
    image.append(img)

    # store the feature
    features = feature.hog(img, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(3, 3), visualise=False, block_norm='L2-Hys')
    digit_data.append(features)


model = []
model_name = []
model.append(tree.DecisionTreeClassifier())  # Decision Tree
model.append(neighbors.KNeighborsClassifier(5, weights='uniform'))  # K-Nearest Neighbor
model.append(naive_bayes.GaussianNB())  # Naive Bayes
model_name.append('CART')
model_name.append('KNN')
model_name.append('NB')

# K-fold cross validation
print("Accuracy on digit recongnition with K-fold cross validation.")
for i in range(len(model)):
    scores = cross_val_score(model[i], digit_data, digit_label, cv=10)
    print("using %s: %0.2f" % (model_name[i], scores.mean()))
print("")


# Split the data randomly to training data and test data (70% / 30% )
# X_train, X_test, y_train, y_test = train_test_split(digit_data, digit_label, test_size=0.3)
# for i in range(len(model)):
#     model[i] = model[i].fit(X_train, y_train)
#     accuracy = (model[i].predict(X_test) == y_test).sum() / len(y_test)
#     print("Accuracy on digit recongnition using %s: %0.2f" % (model_name[i], accuracy))


# save array as image file
counter = np.zeros(10,dtype=np.int)
for i in range(len(digit_label)):
    digit = digit_label[i]
    im = Image.fromarray(image[i]*10)
    im.convert("RGB").save("image/{}-{}.jpg".format(digit,counter[digit]))
    counter[digit] = counter[digit] + 1
