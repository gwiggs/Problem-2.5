# fastapi_app/main.py
from fastapi import FastAPI
from classes.backend_class_example import HelloWorld
app = FastAPI()

@app.get("/")
def hello_world():
    exampleClass = HelloWorld()
    return {"Message": exampleClass.get()}