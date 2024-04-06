import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, jsonify, request
from pymongo import MongoClient
 
# from bson import ObjectId
# # from flask.json import JSONEncoder

# # Define the custom JSON encoder
# class CustomJSONEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, ObjectId):
#             return str(obj)
#         return super(CustomJSONEncoder, self).default(obj)
#         app.json_encoder = JSONEncoder


app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():

    client = MongoClient("mongodb+srv://signatureresourcehub:signature@cluster0.ww1qbms.mongodb.net/")
    db = client["db_project1"]
    collection = db["buses"]
    request_data = request.json

    bus_from = request_data.get("busFrom")
    bus_to = request_data.get("busTo")
    ac_non_ac = request_data.get("acNonAc")
    cost_of=request_data.get("cost")
    query = {"busFrom": bus_from, "busTo": bus_to, "acNonAc": ac_non_ac}

    cursor = collection.find(query)

    df = pd.DataFrame(list(cursor))
    print(df['cost'].unique())
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')


    df = df.dropna(subset=['cost'])
    print(df.head())
#     print(df.head())
    df = df[df['cost'] <= cost_of]
    X = pd.DataFrame(df)[['busFrom', 'busTo', 'acNonAc', 'cost', 'busType']]
    y = pd.DataFrame(df)['busName']
    print(X)
    print(y)
    X = df[['busFrom', 'busTo', 'acNonAc', 'cost', 'busType']]
    X.loc[:, 'cost'] = pd.to_numeric(X['cost'], errors='coerce')
    print(1)
    X['cost'] = X['cost'].replace(',', '').astype(float)
    print(2)
    encoder = LabelEncoder()
    print(3)
    X['busFrom'] = encoder.fit_transform(X['busFrom'])
    X['busTo'] = encoder.fit_transform(X['busTo'])
    X['acNonAc'] = encoder.fit_transform(X['acNonAc'])
    X['busType'] = encoder.fit_transform(X['busType'])
    X.fillna(X.mean(), inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train)

    print(X_train.dtypes)
    print(y_train.dtypes)

    print(y_train.value_counts())

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred_rf = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_rf)
    accuracy_percentage_rf = accuracy * 100
    print("Accuracy:", accuracy_percentage_rf, "%")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    k = 5
    modell = KNeighborsClassifier(n_neighbors=k)
    modell.fit(X_train_scaled, y_train)

    y_pred_kn = modell.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred_kn)

    accuracy_percentage_kn = accuracy * 100
    print("Accuracy:", accuracy_percentage_kn, "%")

    # Convert X and y to JSON format
    X_json = X.to_json(orient='records')
    y_json = y.to_json(orient='records')

    # Printing JSON data
    print("X JSON format:")
    #print(X_json)
    print("\ny JSON format:")
    outside = [{
    "busFrom": "Angamaly",
    "busTo": "Bangalore",
    "acNonAc": "AC",
    "cost": 2700.0,
    "busType": "Sleeper"
}]

    df = pd.DataFrame(outside)

    df = pd.DataFrame(list(outside))
    print(df.head())
    X = pd.DataFrame(df)[['busFrom', 'busTo', 'acNonAc', 'cost', 'busType']]
    X = pd.DataFrame(df)[list(df.columns)]
    print(X)
    X = df[['busFrom', 'busTo', 'acNonAc', 'cost', 'busType']]
    X['cost'] = pd.to_numeric(X['cost'], errors='coerce')  # Convert to numeric, handle errors
    print(1)
    X['cost'] = X['cost'].replace(',', '').astype(float)
    print(2)
    encoder = LabelEncoder()
    print(3)
    X['busFrom'] = encoder.fit_transform(X['busFrom'])
    X['busTo'] = encoder.fit_transform(X['busTo'])
    X['acNonAc'] = encoder.fit_transform(X['acNonAc'])
    X['busType'] = encoder.fit_transform(X['busType'])
    X.fillna(X.mean(), inplace=True)
    X_json = X.to_json(orient='records')
    # Printing JSON data
    print("X JSON format:")
    print(X_json)
    response = {
        "RandomForestClassifier": {},
        # "KNeighborsClassifier": {},
        # "BusesBelowGivenCost": []
    }

    test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    unique_test_data = test_data.drop_duplicates()
    
    # Iterate over unique bus details and make predictions for each
    for idx, row in unique_test_data.iterrows():
        bus_details = row.to_dict()
        bus_name = bus_details.pop('busName')
        
        # Make predictions for RandomForestClassifier
        rf_prediction = y_pred_rf[idx] if bus_name in y_test.values else None

        travel_details = collection.find_one({"busName": bus_name}, 
                                         {"busFrom": 1, "busTo": 1, "startTime": 1, 
                                          "endTime": 1, "acNonAc": 1, "cost": 1, 
                                          "noOfSeats": 1, "busType": 1})
        if travel_details and '_id' in travel_details:
            travel_details['_id'] = str(travel_details['_id'])

    # Add unique bus details and predictions to the response
    response["RandomForestClassifier"][bus_name] = {
        "details": bus_details,
        "prediction": rf_prediction,
        "travel_details": travel_details  # Include specific fields from MongoDB in the response
    }
    response = {"RandomForestClassifier": {}}

    test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    unique_test_data = test_data.drop_duplicates()

    for idx, row in unique_test_data.iterrows():
        bus_details = row.to_dict()
        bus_name = bus_details.pop('busName')

        rf_prediction = y_pred_rf[idx] if bus_name in y_test.values else None
        client = MongoClient('mongodb+srv://signatureresourcehub:signature@cluster0.ww1qbms.mongodb.net/')
        db = client['db_project1']
        collection = db['buses']
        travel_details = collection.find_one({"busName": bus_name}, 
                                            {"busFrom": 1, "busTo": 1, "startTime": 1, 
                                             "endTime": 1, "acNonAc": 1, "cost": 1, 
                                             "noOfSeats": 1, "busType": 1})

        response["RandomForestClassifier"][bus_name] = {
            "details": bus_details,
            "prediction": rf_prediction,
            "travel_details": travel_details
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

# Return the response as JSON
    # return jsonify(response)

        
        # Make predictions for KNeighborsClassifier
        # kn_prediction = y_pred_kn[idx] if bus_name in y_test.values else None
        
        # Add unique bus details and predictions to the response
    #     response["RandomForestClassifier"][bus_name] = {
    #         "details": bus_details,
    #         # "prediction": rf_prediction
    #     }
    #     # response["KNeighborsClassifier"][bus_name] = {
    #     #     "details": bus_details,
    #     #     # "prediction": kn_prediction
    #     # }
    #     # if bus_details['cost'] < cost_of:
    #     #     response["BusesBelowGivenCost"].append({
    #     #         "busName": bus_name,
    #     #         "details": bus_details
    #     #     })
     
    
    # # Return the response as JSON
    # return jsonify(response)
    
    



# if __name__ == '__main__':
     
#  app.run(debug=True)