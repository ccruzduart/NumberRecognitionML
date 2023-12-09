import os
import pickle
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from image_processing import process_image
# from image_processing import process_image


def createStorage():
    BASE_DIR = os.getcwd()
    CLASSIFIER_STORAGE_DIR = os.path.join(BASE_DIR, 'storage')

    # Check if the folder exists
    if not os.path.exists(CLASSIFIER_STORAGE_DIR):
        # If it doesn't exist, create it
        os.makedirs(CLASSIFIER_STORAGE_DIR)

    # Now, CLASSIFIER_STORAGE_DIR points to an existing directory
    CLASSIFIER_STORAGE = os.path.join(CLASSIFIER_STORAGE_DIR, 'classifier.txt')
    return CLASSIFIER_STORAGE


def ClassifierFactory(data, target):
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(data, target)
    return model

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

def updateStorage(classifier):
    with open(createStorage(), 'wb') as in_:
        pickle.dump(classifier, in_)

def PredictDigitService(image_path):
    classifier = getStorage()
    if classifier is None:
        digits = load_digits()
        classifier = ClassifierFactory(digits.data, digits.target)
    updateStorage(classifier)
    x = process_image(image_path)
    if x is None:
        return 0

    prediction = classifier.predict(x)[0]
    return prediction


def getImage():
    while True: 
        image_option = input("Enter the number to be predicted: \n0, 3, 4, 5, 8\n>>")
        if image_option == '0':
            return os.path.join(os.getcwd(), 'Numbers', '0.png')
        elif image_option == '3':
            return os.path.join(os.getcwd(), 'Numbers', '3.png')
        elif image_option == '4':
            return os.path.join(os.getcwd(), 'Numbers', '4.png')
        elif image_option == '5':
            return os.path.join(os.getcwd(), 'Numbers', '5.png')
        elif image_option == '8':
            return os.path.join(os.getcwd(), 'Numbers', '8.png')
        else:
            print('Please enter a valid option\n')

def main():
    storage = getStorage()
    image = getImage()
    prediction = PredictDigitService(image)
    print('Prediction: ' + str(prediction))

if __name__ == "__main__":
    main()