from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('model/energy_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Example: you expect a list of features
    features = data[['country', 'year']]
    
    prediction = model.predict([features])
    output = prediction[0]
    
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
