# 1.  Import libraries
import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import sklearn
from banknoteattributes import BankNote
import pickle

# 2. Create app object
app = FastAPI()

# 3. Load the classifier file
with open('svclassifier.pickle', 'rb') as file:
    classifier = pickle.load(file)


# 4. Index page
@app.get("/")
def index():
    return {"Data": "Go to /note-authenticity page."}


# 5. Make a prediction from passed JSON data
@app.post('/note-authenticity')
def predict_note_authenticity(data: BankNote):
    data = data.dict()
    print(data)
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    predict = classifier.predict([[variance, skewness, curtosis, entropy]])
    return {"prediction": "Fake Note"} if predict[0] > 0.5 else {"prediction": "Bank Note."}


# 6. Run the API
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)