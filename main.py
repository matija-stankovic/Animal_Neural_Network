"""
Matija Stankovic - matt.m.stankovic@gmail.com
Cat or Dog Machine Learning programme
April 2022

"""

from tqdm import tqdm
import numpy as np
import pickle
import random
import cv2
import os

Data_Directory = "PetImages"  # Relative path
Types_Of_Animals = []  # List of all Animals
training_data = []
Size_of_Image = 100

# This for-loop finds all the directories entitled by the type of animal
for path, dir, files in os.walk(Data_Directory):
    for directory in dir:
        Types_Of_Animals.append(directory)


def initialise_training_data():
    for Type_of_Animal in Types_Of_Animals:
        path = os.path.join(Data_Directory, Type_of_Animal)
        class_num = Types_Of_Animals.index(Type_of_Animal)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (Size_of_Image, Size_of_Image))
                training_data.append([new_array, class_num])

            except Exception as e:
                pass


initialise_training_data()

print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, Size_of_Image, Size_of_Image, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
