from class_lib.linear_regression import Predict_Price
import pytest
import os
from config import *

def test_predictclass_init_not0or1():
    with pytest.raises(Exception):
        Predict_Price(2)

def test_predictclass_init_string():
    with pytest.raises(Exception):
        Predict_Price('he')

def test_folder_creations():
    Predict_Price()
    assert os.path.exists(OUTPUT_PATH) == True
    assert os.path.exists(MODELS_PATH) == True

def test_train_pickle_isavailable():
    Predict_Price(0)
    model_pkl_filename = MODELS_PATH + 'boston_price.pkl'
    assert os.path.isfile(model_pkl_filename) == True

def test_predict_generates_error_for_non_2D_input():
    with pytest.raises(Exception):
        Predict_Price(1).predict([0.21124,12.5,7.87,0.0,0.524,5.631,100.0,6.0821,5.0,311.0,15.2,386.63,29.93])

def test_predict_generates_result():
    rst = Predict_Price(1).predict([[0.21124,12.5,7.87,0.0,0.524,5.631,100.0,6.0821,5.0,311.0,15.2,386.63,29.93]])
    assert rst.size != 0