import tensorflow as tf
import cv2
import os

Data_Directory = "PetImages"  # Relative path
Types_Of_Animals = []  # List of all Animals

# This for-loop finds all the directories entitled by the type of animal
for path, dir, files in os.walk(Data_Directory):
    for directory in dir:
        Types_Of_Animals.append(directory)

print(Types_Of_Animals)


def prepare(filepath):
    Image_Size = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (Image_Size, Image_Size))
    return new_array.reshape(-1, Image_Size, Image_Size, 1)


model = tf.keras.models.load_model("Pets.model")

prediction = model.predict([prepare('1.jpg')])
print(prediction)
print(Types_Of_Animals[int(prediction[0][0])])
prediction = model.predict([prepare('img.png')])
print(prediction)
print(Types_Of_Animals[int(prediction[0][0])])