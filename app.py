from flask import Flask, request, render_template
import numpy as np
import pickle

# load trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standardscaler.pkl', 'rb'))     # ✅ corrected
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))    # ✅ corrected

# create flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # get input values from form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])   # ✅ spelling must match HTML
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # feature array
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # scaling
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # model prediction
        prediction = model.predict(final_features)

        # crop dictionary (dataset has labels mapped 1-22)
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
            10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"🌱 {crop} is the best crop to be cultivated right there."
        else:
            result = "❌ Sorry, could not determine the best crop with given data."

    except Exception as e:
        result = f"⚠️ Error: {str(e)}"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
