from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client['image_classification_db']
model_collection = db['model']

with open('model/model.h5', 'rb') as f:
    model_binary = f.read()

model_collection.insert_one({"filename": "cancer_model.h5", "data": model_binary})
print("Cancer Model 1 successfully stored in the database.")

with open('model/Model_2.h5', 'rb') as f:
    model_binary = f.read()

model_collection.insert_one({"filename": "cancer_model_2.h5", "data": model_binary})
print("Cancer Model 2 successfully stored in the database.")
