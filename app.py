from flask import Flask, render_template, request, jsonify
import joblib, os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DB_URI", "sqlite:///posts.db")
# Load saved models
vectorizer = joblib.load('count_vectorizer.joblib')
naive_bayes_model = joblib.load('naive_bayes_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
svm_model = joblib.load('svm_model.joblib')
xgb_model = joblib.load('xgboost_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

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
    class_counts = {nb_prediction: 1, svm_prediction: 1, xgb_prediction: 1}
    final_prediction = max(class_counts, key=class_counts.get)

    return render_template('index.html', question=question, nb_prediction=nb_prediction,
                           svm_prediction=svm_prediction, xgb_prediction=xgb_prediction, final_prediction=final_prediction)

if __name__ == '__main__':
    app.run(debug=True)
