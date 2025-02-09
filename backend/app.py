from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from pymongo import MongoClient
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

client = MongoClient("mongodb://localhost:27017/")
db = client['image_classification_db']
model_collection = db['model']
predictions_collection = db['predictions']
users_collection = db['users']

def load_model(model_filename):
    model_data = model_collection.find_one({"filename": model_filename})
    if not model_data:
        raise ValueError(f"Model {model_filename} not found in the database.")
    with open("temp_model.h5", "wb") as f:
        f.write(model_data["data"])
    model = tf.keras.models.load_model("temp_model.h5")
    print(f"{model_filename} loaded successfully.")
    return model

# --------------------------------------------------------------------------------
# -------------------------- Loading Models --------------------------------------
# --------------------------------------------------------------------------------

model_1 = load_model("cancer_model.h5")
model_2 = load_model("cancer_model_2.h5")

# --------------------------------------------------------------------------------
# -------------------------- Data Preprocessing ----------------------------------
# --------------------------------------------------------------------------------

def preprocess_image_model_2(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((8, 8))
    image_array = np.array(image) / 255.0
    processed_image = np.expand_dims(image_array, axis=0)
    return processed_image

def preprocess_image_model_1(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    processed_image = np.expand_dims(image_array, axis=0)
    return processed_image

# --------------------------------------------------------------------------------
# -------------------------------- GET Methods -----------------------------------
# --------------------------------------------------------------------------------

@app.get("/")
def serve_frontend():
    return RedirectResponse(url="/login")

@app.get("/index")
def serve_index():
    return FileResponse("frontend/index.html")

@app.get("/login")
def login_page():
    return FileResponse("frontend/login.html")

@app.get("/register")
def register_page():
    return FileResponse("frontend/register.html")

# --------------------------------------------------------------------------------
# -------------------------------- POST Methods -----------------------------------
# --------------------------------------------------------------------------------

@app.post("/predict/model_1/")
async def predict_model_1(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image_model_1(contents)
        prediction = model_1.predict(image)
        result = "Cancer Detected" if prediction[0][0] < 0.5 else "No Cancer"

        predictions_collection.insert_one({
            "filename": file.filename,
            "prediction": result,
            "image": contents
        })

        if result == "Cancer Detected":
            return HTMLResponse(content=open("./frontend/cancer.html").read())
        else:
            return HTMLResponse(content=open("./frontend/noCancer.html").read())
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

@app.post("/predict/model_2/") 
async def predict_model_2(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image_model_2(contents)
        prediction = model_2.predict(image)
        result = "Cancer Detected" if prediction[0][0] < 0.5 else "No Cancer"

        predictions_collection.insert_one({
            "filename": file.filename,
            "prediction": result,
            "image": contents
        })

        if result == "Cancer Detected":
            return HTMLResponse(content=open("./frontend/cancer.html").read())
        else:
            return HTMLResponse(content=open("./frontend/noCancer.html").read())
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    existing_user = users_collection.find_one({"username": username})
    if existing_user:
        return HTMLResponse(content="<h2>Username already exists. Please choose a different username.</h2>", status_code=400)

    users_collection.insert_one({
        "username": username,
        "password": password
    })

    return RedirectResponse(url="/login", status_code=303)

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    existing_user = users_collection.find_one({"username": username, "password": password})
    if existing_user:
        return RedirectResponse(url="/index", status_code=303)

    return RedirectResponse(url="/register", status_code=303)
