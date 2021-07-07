from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os, sys
sys.path.insert(1, os.path.abspath('.'))

from config import *


#print(OUTPUT_PATH)

class Predict_Price:
    def __init__(self, USE_PRETRAINED_MODELS:int = 0) -> None:

        if not USE_PRETRAINED_MODELS in (0,1):
            raise ValueError("Instance 0 or 1 required")

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)

        self.__data = load_boston()
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(self.__data.data, self.__data.target)
        
        if USE_PRETRAINED_MODELS == 0:
            self.__train()


    @property
    def x_train(self) -> np.array:
        return self.__x_train

    @property
    def x_test(self) -> np.array:
        return self.__x_test

    @property
    def y_train(self)-> np.array:
        return self.__y_train

    @property
    def y_test(self) -> np.array:
        return self.__y_test


    def __train(self) -> np.array:
        clf = LinearRegression()
        clf.fit(self.__x_train, self.__y_train)
        predicted = clf.predict(self.__x_test)
        expected = self.__y_test
        accuracy = self.__accuracy(predicted, expected)
        print ("RMS: %r " % accuracy)
        print (f"Mean Absolute Error: {mean_absolute_error(expected, predicted)}")
        model_pkl_filename = MODELS_PATH + 'boston_price.pkl'
        with open(model_pkl_filename, "wb") as f:
            pickle.dump(clf, f)
        self.__plot_train_prediction(predicted, expected)

    def predict(self, input_param: np.array):
        val = self.__check_array_validation(input_param)
        model_pkl_filename = MODELS_PATH + 'boston_price.pkl'
        with open(model_pkl_filename, "rb") as f:
            clf = pickle.load(f)
            predictions = clf.predict(input_param)

        return np.array_str(predictions)

    def __accuracy(self, predicted: np.array, expected:np.array) -> float:
        return np.sqrt(np.mean((predicted - expected) ** 2))

    def __plot_train_prediction(self, predicted: np.array, expected:np.array) -> None:
        plt.figure(figsize=(4, 3))
        plt.scatter(expected, predicted)
        plt.plot([0, 50], [0, 50], '--k')
        plt.axis('tight')
        plt.xlabel('True price ($1000s)')
        plt.ylabel('Predicted price ($1000s)')
        plt.tight_layout()
        pred_path = OUTPUT_PATH + 'boston_price_prediction.png'
        plt.savefig(pred_path)


    def __check_array_validation(self, input_param: np.array):
        val = np.asarray(input_param)
        if len(val.shape) != 2:
            raise ValueError (f"Expected 2D array for the prediction, got {len(val.shape)}D array")
        return val



