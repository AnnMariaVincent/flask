import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, jsonify, request
from pymongo import MongoClient

app = Flask(__name__)
encoders = {}

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
    bus_type=request_data.get("busType")
    query = {"busFrom": bus_from, "busTo": bus_to, "acNonAc": ac_non_ac,"busType":bus_type}
    cursor = collection.find(query)
    print(cursor)
    df = pd.DataFrame(list(cursor))
    print(df['cost'].unique())
    # Convert the 'cost' column to float, handling non-string values
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    # Drop rows with NaN values in the 'cost' column
    df = df.dropna(subset=['cost'])
    #set data for training
    X = pd.DataFrame(df)[['busFrom', 'busTo', 'acNonAc', 'cost', 'busType']]
    y = pd.DataFrame(df)['busName']
    X = df[['busFrom', 'busTo', 'acNonAc', 'cost', 'busType']]
    X.loc[:, 'cost'] = pd.to_numeric(X['cost'], errors='coerce')
    X['cost'] = X['cost'].replace(',', '').astype(float)

    #set label encoding
    encoder = LabelEncoder()
    X['busFrom'] = encoder.fit_transform(X['busFrom'])
    X['busTo'] = encoder.fit_transform(X['busTo'])
    X['acNonAc'] = encoder.fit_transform(X['acNonAc'])
    X['busType'] = encoder.fit_transform(X['busType'])
    X.fillna(X.mean(), inplace=True)
    #set train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #create model in random forest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred_rf = model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_rf)
    accuracy_percentage_rf = accuracy * 100
    print("Accuracy:", accuracy_percentage_rf, "%")

    
    #create model in knn
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

    outside=[{
    "busFrom":bus_from,
    "busTo":bus_to,
    "acNonAc":ac_non_ac,
    "cost":cost_of,
    "busType":bus_type
    }]
    print(outside)
    dfn = pd.DataFrame(list(outside))
    print(dfn.head())
    Xn = pd.DataFrame(dfn)[['busFrom', 'busTo', 'acNonAc', 'cost', 'busType']]
    Xn = dfn[['busFrom', 'busTo', 'acNonAc', 'cost', 'busType']]
    Xn['cost'] = pd.to_numeric(Xn['cost'], errors='coerce')  # Convert to numeric, handle errors
    print(1)
    Xn['cost'] = Xn['cost'].replace(',', '').astype(float)
    print(2)
    encoder = LabelEncoder()
    print(3)
    Xn['busFrom'] = encoder.fit_transform(Xn['busFrom'])
    Xn['busTo'] = encoder.fit_transform(Xn['busTo'])
    Xn['acNonAc'] = encoder.fit_transform(Xn['acNonAc'])
    Xn['busType'] = encoder.fit_transform(Xn['busType'])
    Xn.fillna(Xn.mean(), inplace=True)
    X_json = Xn.to_json(orient='records')
    # Printing JSON data
    print("X JSON format:")
    print(X_json)
    data = X_json
    data_list = json.loads(data)
    bus_from = data_list[0]["busFrom"]
    prediction=[]
    if accuracy_percentage_kn>accuracy_percentage_rf:
        pred_op = modell.predict([[data_list[0]["busFrom"],data_list[0]["busTo"],data_list[0]["acNonAc"],data_list[0]["cost"],data_list[0]["busType"]]])
        print("knn",pred_op)
        for i in pred_op:
           prediction.append(i)
    else:
        pred_op = model.predict([[data_list[0]["busFrom"],data_list[0]["busTo"],data_list[0]["acNonAc"],data_list[0]["cost"],data_list[0]["busType"]]])
        print("rf",pred_op)
        for i in pred_op:
           prediction.append(i)
    results = []
    for i in prediction:
        query = {"busFrom":request_data.get("busFrom"), "busTo": request_data.get("busTo"), "acNonAc": request_data.get("acNonAc"),"busType":request_data.get("busType"),"busName":i}
        cursor = collection.find(query)
        for document in cursor:
            document['_id'] = str(document['_id'])
            results.append(document)
    if results:
        return jsonify(results), 200
    else:
        return jsonify({"message": "No buses found for the given criteria"}), 404
    

if __name__ == '__main__':
     
 app.run(debug=True)