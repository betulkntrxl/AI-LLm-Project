from flask import Flask, request, jsonify, session
import numpy as np
from tensorflow.keras.models import load_model
import os
import pandas as pd
import requests
from flask_cors import CORS 
import json

app = Flask(__name__)
CORS(app)  


api_key = os.getenv('OPENAI_API_KEY')
app.secret_key = 'mF7mVDQUoXOMZwp25GP04f1Y5F4ZRowt' 


heart_disease_model = load_model('heart_disease_model.keras')
lung_cancer_model = load_model('lung_cancer_model.keras')
diabetes_model = load_model('diabetes_model.keras')

def initialize_conversation():
    if 'conversation_history' not in session:
        session['conversation_history'] = []

diabetes_mapping = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "blood pressure": "BloodPressure",
    "skin thickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "diabetes pedigree function": "DiabetesPedigreeFunction",
    "age": "Age"
}

heart_mapping = {
    "age": "age",
    "gender": "sex",
    "chest pain": "cp",
    "resting blood pressure": "trestbps",
    "cholesterol": "chol",
    "fasting blood sugar": "fbs",
    "resting ecg": "restecg",
    "max heart rate": "thalach",
    "exercise induced angina": "exang",
    "oldpeak": "oldpeak",
    "slope": "slope",
    "number of major vessels": "ca",
    "thal": "thal"
}

lung_mapping = {
    "age": "Age",
    "gender": "Gender",
    "air pollution": "Air Pollution",
    "alcohol use": "Alcohol use",
    "dust allergy": "Dust Allergy",
    "occupational hazards": "OccuPational Hazards",
    "genetic risk": "Genetic Risk",
    "chronic lung disease": "chronic Lung Disease",
    "balanced diet": "Balanced Diet",
    "obesity": "Obesity",
    "smoking": "Smoking",
    "passive smoker": "Passive Smoker",
    "chest pain": "Chest Pain",
    "coughing of blood": "Coughing of Blood",
    "fatigue": "Fatigue",
    "weight loss": "Weight Loss",
    "shortness of breath": "Shortness of Breath",
    "wheezing": "Wheezing",
    "swallowing difficulty": "Swallowing Difficulty",
    "clubbing of finger nails": "Clubbing of Finger Nails",
    "frequent cold": "Frequent Cold",
    "dry cough": "Dry Cough",
    "snoring": "Snoring"
}

def compute_column_means(csv_path):
    df = pd.read_csv(csv_path)
    means = df.min(numeric_only=True).to_dict()
    return means

heart_feature_means = compute_column_means('heart.csv')
lung_feature_means = compute_column_means('lung.csv')
diabetes_feature_means = compute_column_means('diabetes.csv')

def query_chatgpt_for_symptoms(input_text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": (
                "You are an assistant that extracts medical symptoms and relevant health details from text and formats them as a JSON object. "
                "Additionally, you should identify the language of the input text and include it as a 'language' key in the JSON response.\n"
                "The JSON should contain specific keys with numerical values, using the following guidelines: \n"
                "- 'age': numerical, representing the patient's age in years. (Min = 14, Max = 80)\n"
                "- 'gender': '0' for male and '1' for female.\n"
                "- 'chest pain': numerical, representing chest pain type. (Min = 0, Max = 3)\n"
                "- 'resting blood pressure': numerical, representing resting blood pressure in mm Hg. (Min = 94, Max = 200)\n"
                "- 'cholesterol': numerical, representing serum cholesterol in mg/dl. (Min = 126, Max = 564)\n"
                "- 'fasting blood sugar': binary, '0' if fasting blood sugar < 120 mg/dl, '1' otherwise. (Min = 0, Max = 1)\n"
                "- 'resting ecg': numerical, representing resting electrocardiographic results. (Min = 0, Max = 2)\n"
                "- 'max heart rate': numerical, representing maximum heart rate achieved. (Min = 71, Max = 202)\n"
                "- 'exercise induced angina': binary, '0' = no, '1' = yes. (Min = 0, Max = 1)\n"
                "- 'oldpeak': numerical, representing ST depression induced by exercise relative to rest. (Min = 0.0, Max = 6.2)\n"
                "- 'slope': numerical, representing the slope of the peak exercise ST segment. (Min = 0, Max = 2)\n"
                "- 'number of major vessels': numerical, representing number of major vessels colored by fluoroscopy. (Min = 0, Max = 4)\n"
                "- 'thal': categorical, representing thalassemia. (Min = 0, Max = 3)\n"
                "- 'air pollution': numerical, representing the level of air pollution exposure. (Min = 1, Max = 8)\n"
                "- 'alcohol use': numerical, representing the level of alcohol use. (Min = 1, Max = 8)\n"
                "- 'dust allergy': numerical, representing the severity of dust allergy. (Min = 1, Max = 8)\n"
                "- 'occupational hazards': numerical, representing exposure to occupational hazards. (Min = 1, Max = 8)\n"
                "- 'genetic risk': numerical, representing the genetic risk factor. (Min = 1, Max = 7)\n"
                "- 'chronic lung disease': numerical, indicating the presence of chronic lung disease. (Min = 1, Max = 7)\n"
                "- 'balanced diet': numerical, representing adherence to a balanced diet. (Min = 1, Max = 7)\n"
                "- 'obesity': numerical, representing the level of obesity. (Min = 1, Max = 7)\n"
                "- 'smoking': numerical, indicating smoking status. (Min = 1, Max = 8)\n"
                "- 'passive smoker': numerical, indicating exposure to second-hand smoke. (Min = 1, Max = 8)\n"
                "- 'coughing of blood': numerical, indicating the frequency of coughing blood. (Min = 1, Max = 9)\n"
                "- 'fatigue': numerical, indicating the level of fatigue. (Min = 1, Max = 9)\n"
                "- 'weight loss': numerical, indicating the extent of weight loss. (Min = 1, Max = 8)\n"
                "- 'shortness of breath': numerical, indicating the severity of shortness of breath. (Min = 1, Max = 9)\n"
                "- 'wheezing': numerical, indicating the severity of wheezing. (Min = 1, Max = 8)\n"
                "- 'swallowing difficulty': numerical, indicating the difficulty in swallowing. (Min = 1, Max = 8)\n"
                "- 'clubbing of finger nails': numerical, indicating the presence of clubbing of fingernails. (Min = 1, Max = 9)\n"
                "- 'frequent cold': numerical, indicating the frequency of cold. (Min = 1, Max = 7)\n"
                "- 'dry cough': numerical, indicating the presence of a dry cough. (Min = 1, Max = 7)\n"
                "- 'snoring': numerical, indicating the frequency of snoring. (Min = 1, Max = 7)\n"
                "- 'pregnancies': numerical, representing the number of pregnancies. (Min = 0, Max = 17)\n"
                "- 'glucose': numerical, representing the plasma glucose concentration. (Min = 0, Max = 300)\n"
                "- 'blood pressure': numerical, representing the diastolic blood pressure (mm Hg). (Min = 0, Max = 122)\n"
                "- 'skin thickness': numerical, representing the triceps skin fold thickness (mm). (Min = 0, Max = 99)\n"
                "- 'insulin': numerical, representing the 2-Hour serum insulin (mu U/ml). (Min = 0, Max = 846)\n"
                "- 'bmi': numerical, representing the body mass index (weight in kg/(height in m)^2). (Min = 0.0, Max = 67.1)\n"
                "- 'diabetes pedigree function': numerical, representing the diabetes pedigree function. (Min = 0.078, Max = 2.42)\n"
                "- 'age': numerical, representing the patient's age in years. (Min = 21, Max = 81)\n"
                "- 'language': the language of the input text (e.g., 'en', 'es', 'fr').\n"
                "If any data is missing or unknown, use 'null'. Ensure no key is missing and values are consistent with medical contexts."
            )},
            {"role": "user", "content": input_text}
        ],
        "temperature": 0.3
    }

    print("calling")
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    print(response)
    if response.status_code == 200:
        json_data = response.json()['choices'][0]['message']['content']
        
        try:
            json_data = json_data.strip()
            
            if json_data.startswith("```json"):
                json_data = json_data[7:]  
            if json_data.endswith("```"):
                json_data = json_data[:-3]  
            parsed_data = json.loads(json_data)
            print(parsed_data)
            
            language = parsed_data.get('language', 'en')  
            return parsed_data, language
        except json.JSONDecodeError:
            print("JSON Format error.")
            return {}, 'en'
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return {}, 'en'

def replace_none_with_mean(features, feature_mapping, feature_means):
    for i, feature in enumerate(features):
        if feature is None:
            csv_column_name = feature_mapping.get(list(feature_mapping.keys())[i])
            if csv_column_name in feature_means:
                features[i] = feature_means[csv_column_name]
            else:
                features[i] = 0  
    return features

def prepare_features_for_models(symptoms_json):
    heart_features = [
        symptoms_json.get("age", None),
        symptoms_json.get("gender", None),
        symptoms_json.get("chest pain", None),
        symptoms_json.get("resting blood pressure", None),
        symptoms_json.get("cholesterol", None),
        symptoms_json.get("fasting blood sugar", None),
        symptoms_json.get("resting ecg", None),
        symptoms_json.get("max heart rate", None),
        symptoms_json.get("exercise induced angina", None),
        symptoms_json.get("oldpeak", None),
        symptoms_json.get("slope", None),
        symptoms_json.get("number of major vessels", None),
        symptoms_json.get("thal", None)
    ]

    lung_features = [
        symptoms_json.get("age", None),
        symptoms_json.get("gender", None),
        symptoms_json.get("air pollution", None),
        symptoms_json.get("alcohol use", None),
        symptoms_json.get("dust allergy", None),
        symptoms_json.get("occupational hazards", None),
        symptoms_json.get("genetic risk", None),
        symptoms_json.get("chronic lung disease", None),
        symptoms_json.get("balanced diet", None),
        symptoms_json.get("obesity", None),
        symptoms_json.get("smoking", None),
        symptoms_json.get("passive smoker", None),
        symptoms_json.get("chest pain", None),
        symptoms_json.get("coughing of blood", None),
        symptoms_json.get("fatigue", None),
        symptoms_json.get("weight loss", None),
        symptoms_json.get("shortness of breath", None),
        symptoms_json.get("wheezing", None),
        symptoms_json.get("swallowing difficulty", None),
        symptoms_json.get("clubbing of finger nails", None),
        symptoms_json.get("frequent cold", None),
        symptoms_json.get("dry cough", None),
        symptoms_json.get("snoring", None)
    ]

    diabetes_features = [
        symptoms_json.get("pregnancies", None),
        symptoms_json.get("glucose", None),
        symptoms_json.get("blood pressure", None),
        symptoms_json.get("skin thickness", None),
        symptoms_json.get("insulin", None),
        symptoms_json.get("bmi", None),  
        symptoms_json.get("diabetes pedigree function", None), 
        symptoms_json.get("age", None)
    ]

    
    heart_features = replace_none_with_mean(heart_features, heart_mapping, heart_feature_means)
    lung_features = replace_none_with_mean(lung_features, lung_mapping, lung_feature_means)
    diabetes_features = replace_none_with_mean(diabetes_features, diabetes_mapping, diabetes_feature_means)

    print("Heart Features:", heart_features)
    print("Lung Features:", lung_features)
    print("Diabetes Features:", diabetes_features)

    return heart_features, lung_features, diabetes_features

def predict_heart_disease_risk(features, model):
    
    input_data = np.array([features], dtype=float)
    prediction = model.predict(input_data)
    print("Heart disease prediction:", prediction)
    
    return float(prediction[0][0])  

def predict_lung_cancer_risk(features, model):
    
    input_data = np.array([features], dtype=float)
    prediction = model.predict(input_data)
    print("Lung cancer prediction:", prediction)
    
    risk_levels = ['Low', 'Medium', 'High']
    predicted_index = np.argmax(prediction[0])
    return risk_levels[predicted_index]  

def predict_diabetes_risk(features, model):
    
    input_data = np.array([features], dtype=float)
    prediction = model.predict(input_data)
    print("Diabetes prediction:", prediction)
    return float(prediction[0][0])  

def generate_explanation_with_chatgpt(heart_risk, lung_risk, diabetes_risk, target_language="en"):
    
    message = (
        f"Please only explain these risks to the patient in language {target_language}."
        f"The patient has a heart disease risk of {heart_risk:.2f}, "
        f"a lung cancer risk of {lung_risk}, and a diabetes risk of {diabetes_risk:.2f}. "
        f"Do not provide numeric values in explanation. "
    )
    print(message)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that provides explanations about health risks based on provided risk scores."},
            {"role": "user", "content": message}
        ],
        "temperature": 0.7
    }

    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    
    if response.status_code == 200:
        explanation = response.json()['choices'][0]['message']['content']
        return explanation
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "There was an error processing your request."

@app.route('/docbot', methods=['POST'])
def docbot():
    
    user_input = request.json.get('input_text', '')
    symptoms_json, detected_language = query_chatgpt_for_symptoms(user_input)
    heart_features, lung_features, diabetes_features = prepare_features_for_models(symptoms_json)
    heart_risk_prediction = heart_disease_model.predict(np.array([heart_features], dtype=float))
    heart_risk = float(heart_risk_prediction[0][0])
    
    lung_risk_prediction = lung_cancer_model.predict(np.array([lung_features], dtype=float))
    lung_risk_index = np.argmax(lung_risk_prediction[0])
    lung_risk = ['low', 'medium', 'high'][lung_risk_index]  
    
    diabetes_risk_prediction = diabetes_model.predict(np.array([diabetes_features], dtype=float))
    diabetes_risk = float(diabetes_risk_prediction[0][0])
    
    print("Heart Risk Prediction: ",heart_risk_prediction)
    print("Lung Risk Prediction: ",lung_risk_prediction)
    print("Diabetes Risk Prediction: ",diabetes_risk_prediction)
    explanation = generate_explanation_with_chatgpt(heart_risk, lung_risk, diabetes_risk, detected_language)
    
    
    return jsonify({
        'explanation': explanation,
        'heart_risk': heart_risk,
        'diabetes_risk': diabetes_risk,
        'lung_risk': lung_risk
    })
def query_chatgpt_for_conversation(input_text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    conversation_history = session.get('conversation_history', [])
    
    if len(conversation_history) == 0:
        conversation_history.append({
    "role": "system",
    "content": (
        "You are a helpful assistant that provides medical advice and guidance based on the user's symptoms or health-related questions. "
        "Do not discuss non-medical topics. "
        "When formatting your response, use HTML entities and elements to represent special characters and formatting: "
        "- Use '&nbsp;' for non-breaking spaces. "
        "- Use '&lt;' for the less-than sign and '&gt;' for the greater-than sign. "
        "- Use '&amp;' for the ampersand symbol. "
        "- Use '<ul>' and '<li>' tags for creating lists. "
        "- Use '<br>' for line breaks. "
        "Provide answers strictly in HTML-safe characters where possible."
    )
})

    
    conversation_history.append({"role": "user", "content": input_text})

    data = {
        "model": "gpt-4o-mini",
        "messages": conversation_history,
        "temperature": 0.7
    }

    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    if response.status_code == 200:
        chat_response = response.json()['choices'][0]['message']['content']
        session['conversation_history'].append({"role": "assistant", "content": chat_response})
        return chat_response
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "There was an error processing your request."

@app.route('/chatbot', methods=['POST'])
def chatbot():
    initialize_conversation()

    user_input = request.json.get('input_text', '')
    chat_response = query_chatgpt_for_conversation(user_input)

    return jsonify({'response': chat_response})

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    session.pop('conversation_history', None)
    return jsonify({'status': 'Conversation reset.'})


if __name__ == '__main__':
    app.run(debug=True)
