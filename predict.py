#!/usr/bin/env python3
#
import os
import pickle
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from image_processing import process_image

# Creates a storage directory file and a classifier text file
# returns the path to the classifier file
def createStorage():
    BASE_DIR = os.getcwd()
    CLASSIFIER_STORAGE_DIR = os.path.join(BASE_DIR, 'storage')

    if not os.path.exists(CLASSIFIER_STORAGE_DIR):
        os.makedirs(CLASSIFIER_STORAGE_DIR)

    CLASSIFIER_STORAGE = os.path.join(CLASSIFIER_STORAGE_DIR, 'classifier.txt')
    return CLASSIFIER_STORAGE

# Creates and trains a K-nearest neighbors model
# returns the trained model
def ClassifierFactory(data, target):
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(data, target)
    return model

# Gets the contents of the classifier file
# returns the input of the file if is not empty
# if empty or not found returns none
def getStorage():
        with open(createStorage(), 'wb') as out:
            try:
                classifier_str = out.read()
                if classifier_str != '':
                    return pickle.loads(classifier_str)
                else:
                    return None
            except Exception:
                return None

# updates the classifier file with a trained model
def updateStorage(model):
    with open(createStorage(), 'wb') as in_:
        pickle.dump(model, in_)

# predicts the selected image
# returns prediction
def PredictDigitService(image_path):
    model = getStorage()
    if model is None:
        digits = load_digits()
        model = ClassifierFactory(digits.data, digits.target)
    updateStorage(model)
    input_image = process_image(image_path)
    if input_image is None:
        return 0

    prediction = model.predict(input_image)[0]
    return prediction

# ask user to select a number to be predicted
# returns the path to the image of the selected number
def getImage():
    while True: 
        image_option = input("Enter the number to be predicted: \n0, 3, 4, 5, 8\n>> ")
        if image_option == '0':
            print('Selected option: '+image_option)
            return os.path.join(os.getcwd(), 'Numbers', '0.png')
        elif image_option == '3':
            print('Selected option: '+image_option)
            return os.path.join(os.getcwd(), 'Numbers', '3.png')
        elif image_option == '4':
            print('Selected option: '+image_option)
            return os.path.join(os.getcwd(), 'Numbers', '4.png')
        elif image_option == '5':
            print('Selected option: '+image_option)
            return os.path.join(os.getcwd(), 'Numbers', '5.png')
        elif image_option == '8':
            print('Selected option: '+image_option)
            return os.path.join(os.getcwd(), 'Numbers', '8.png')
        else:
            print('Please enter a valid option\n')

def main():
    image = getImage()
    prediction = PredictDigitService(image)
    print('Prediction: ' + str(prediction))

if __name__ == "__main__":
    main()