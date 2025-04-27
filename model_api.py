from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('model/energy_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract features from the incoming data
    country = data.get('country')
    year = data.get('year')
    
    # Make sure the features are in the right format for the model
    features = [country, year]  # Assuming the model expects this format
    
    # Make prediction
    prediction = model.predict([features])  # Model prediction expects a 2D array-like
    output = prediction[0]  # Get the first (and presumably only) prediction
    
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
