from flask import Flask
from flask_cors import CORS,cross_origin
app=Flask(__name__)
CORS(app)
@app.route('/')
def HomePage():
    return{
         "Name":"Nirmal",
        "Age":"20"
    } 

