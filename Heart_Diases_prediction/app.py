from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Prediction function
def predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes,
            totChol, sysBP, diaBP, BMI, heartRate, glucose):
    
    # DÜZELTME: HTML'den zaten "1" veya "0" stringi geliyor. 
    # Bunları direkt int() ile sayıya çevirmek yeterli, karmaşık encode işlemlerine gerek yok.
    features = np.array([[
        int(male), int(age), int(currentSmoker), float(cigsPerDay), 
        int(BPMeds), int(prevalentStroke), int(prevalentHyp), int(diabetes), 
        float(totChol), float(sysBP), float(diaBP), float(BMI), 
        float(heartRate), float(glucose)
    ]])

    # Scale the features
    scaled_features = scaler.transform(features)

    # Predict using the model
    result = model.predict(scaled_features)

    return result[0]


# Routes
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if request.method == 'POST':
        try:
            # HTML formundan gelen tüm verileri güvenli bir şekilde çekiyoruz
            # input alanları boş kalma ihtimaline karşı bir hata fırlatmaması için default değer veya try-except iyi olur
            male = request.form['male']
            age = request.form['age']
            currentSmoker = request.form['currentSmoker']
            cigsPerDay = request.form['cigsPerDay']
            BPMeds = request.form['BPMeds']
            prevalentStroke = request.form['prevalentStroke']
            prevalentHyp = request.form['prevalentHyp']
            diabetes = request.form['diabetes']
            totChol = request.form['totChol']
            sysBP = request.form['sysBP']
            diaBP = request.form['diaBP']
            BMI = request.form['BMI']
            heartRate = request.form['heartRate']
            glucose = request.form['glucose']

            # Tahmin fonksiyonunu çağırıyoruz
            prediction = predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, 
                                 prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)
            
            prediction_text = "The Patient has Heart Disease" if prediction == 1 else "The Patient has No Heart Disease"
            
        except ValueError:
            # Eğer kullanıcı formda bir alanı boş bırakırsa sayfanın çökmesini engeller
            prediction_text = "Error: Please fill in all the fields in the form correctly."

        return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
