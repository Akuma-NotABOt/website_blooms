from flask import Flask, render_template, request, jsonify
import joblib, os
from collections import Counter
from make_csv import get_csv, scorecheck
import pandas as pd
from verbs import sig_verbs
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DB_URI", "sqlite:///posts.db")
# Load saved models
"""vectorizer = joblib.load('count_vectorizer.joblib')
naive_bayes_model = joblib.load('naive_bayes_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
svm_model = joblib.load('svm_model.joblib')
xgb_model = joblib.load('xgboost_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')"""
vectorizer = joblib.load(r"D:\Codes\BTP II\newest\website_blooms\count_vectorizer.joblib")
naive_bayes_model = joblib.load(r'D:\Codes\BTP II\newest\website_blooms\naive_bayes_model.joblib')
tfidf_vectorizer = joblib.load(r'D:\Codes\BTP II\newest\website_blooms\tfidf_vectorizer.joblib')
svm_model = joblib.load(r'D:\Codes\BTP II\newest\website_blooms\svm_model.joblib')
xgb_model = joblib.load(r'D:\Codes\BTP II\newest\website_blooms\xgboost_model.joblib')
label_encoder = joblib.load(r"D:\Codes\BTP II\newest\website_blooms\label_encoder.joblib")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    question = request.form['question']
    
    # Predict using Naive Bayes model
    nb_prediction = naive_bayes_model.predict(vectorizer.transform([question]))[0]

    # Predict using SVM model
    svm_prediction = svm_model.predict(tfidf_vectorizer.transform([question]))[0]

    # Predict using XGBoost model
    xgb_prediction_encoded = xgb_model.predict(tfidf_vectorizer.transform([question]))
    xgb_prediction = label_encoder.inverse_transform(xgb_prediction_encoded)[0]
    
   # Count the occurrences of each predicted class
    predictions = [nb_prediction, svm_prediction, xgb_prediction]
    class_counts = Counter(predictions)

    # Find the class with the highest count
    final_prediction = max(class_counts, key=class_counts.get)

    return render_template('index.html', question=question, nb_prediction=nb_prediction,
                           svm_prediction=svm_prediction, xgb_prediction=xgb_prediction, final_prediction=final_prediction)

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded PDF file
        pdf_file = request.files['pdf_file']

        # Save the uploaded PDF file to a temporary location
        pdf_file_path = 'temp.pdf'
        pdf_file.save(pdf_file_path)

        get_csv(pdf_file_path)
        df = pd.read_csv('questions.csv')

        # return jsonify({'message': 'CSV file generated successfully'})
        return render_template('upload.html', dataframe = df, score = (scorecheck(df)*100/6), verbs= sig_verbs(df)) 
    else:
        # Handle GET request (if needed)
        return render_template('upload.html')  # Render the upload.html page

if __name__ == '__main__':
    app.run(debug=True)
