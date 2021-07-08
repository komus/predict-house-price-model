# Predicting House Prices

This project is about creating an endpoint that predicts house prices based on inputted features. 


## Endpoint
The endpoint `/predict` accepts payload of the following structure.
``` http
POST https://lynkideas-price.herokuapp.com/api/predict/
```
|Parameter |  Type         |  Description           |
|----------|:-------------:|-----------------------:|
| `inputs` |  `2 D array`  | Features to be predicted of shape (13,2)|


## Responses
sucesss response
``` javascript
{
  "status_code": 100,
  "status_message": "Okay",
  "data": "[12.32062341 18.99951441]"
}
```

The `status_code` attribute contains a message commonly used to indicate the inner status of the executed endpoint

The `status_message` The message corresponding to the `status code`.

The `data` the result of the payload.

## Example Payload
Calling the endpoint `/api/predict/` with a 2x2 array. 
``` javascript
{
    "inputs": [[0.21124,12.5,7.87,0.0,0.524,5.631,100.0,6.0821,5.0,311.0,15.2,386.63,29.93],
               [0.17004,12.5,7.87,0.0,0.524,6.004,85.9,6.5921,5.0,311.0,15.2,386.71,17.1] ]
    
}
```
