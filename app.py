from flask import Flask,request
from flask_restful import Resource, Api
from config import API_Status, API_Status_Message
import numpy as np
from class_lib.linear_regression import Predict_Price


app = Flask(__name__)
api = Api(app)

class Boston_Price(Resource):
    def post(self):
        data = request.get_json(force=True)
        if "inputs" in data :
            try:
                cls = Predict_Price(1).predict(data['inputs'])
                return {'status_code':  API_Status.OKAY,
                    'status_message':API_Status_Message.LIST,
                    'data': cls}, 200
            except Exception as ex:
                return {'status_code':  API_Status.BAD_TYPE,
                    'status_message': ex.args,
                    'data': None}, 400
        else:
            return {'status_code':  API_Status.BAD_PAYLOAD,
                    'status_message':API_Status_Message.BAD_PAYLOAD,
                    'data': None}, 406




api.add_resource(Boston_Price, '/api/predict/')
app.run()








