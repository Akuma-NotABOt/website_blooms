from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import os, joblib
from collections import Counter

api_key = os.getenv("api_key")
vectorizer = joblib.load(r"BTP II\newest\website_blooms\count_vectorizer.joblib")
naive_bayes_model = joblib.load(r"BTP II\newest\website_blooms\naive_bayes_model.joblib")
tfidf_vectorizer = joblib.load(r"BTP II\newest\website_blooms\tfidf_vectorizer.joblib")
svm_model = joblib.load(r"BTP II\newest\website_blooms\svm_model.joblib")
xgb_model = joblib.load(r"BTP II\newest\website_blooms\xgboost_model.joblib")
label_encoder = joblib.load(r'BTP II\newest\website_blooms\label_encoder.joblib')

def extract_questions_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    questions = []
    for page in pages:
        response = get_chat_completion("'" + str(page) + "'\n\nList all the questions in the extract of question paper. Ignore all options")
        questions.extend(response.split('\n'))
    # List comprehension to filter questions
    filtered_questions = [question for question in questions if len(question) >= 6]

    return filtered_questions

client = OpenAI(
    # This is the default and can be omitted
  api_key = api_key,
)
 
def get_chat_completion(prompt, model="gpt-3.5-turbo"):
  
   # Creating a message as required by the API
   messages = [{"role": "user", "content": prompt}]
  
   # Calling the ChatCompletion API
   response = client.chat.completions.create(
       model=model,
       messages=messages,
       temperature=0.5,
   )

   # Returning the extracted response
   return response.choices[0].message.content

def predict_questions(questions):
    predictions = []
    for question in questions:
        # Predict using Naive Bayes model
        nb_prediction = naive_bayes_model.predict(vectorizer.transform([question]))[0]

        # Predict using SVM model
        svm_prediction = svm_model.predict(tfidf_vectorizer.transform([question]))[0]

        # Predict using XGBoost model
        xgb_prediction_encoded = xgb_model.predict(tfidf_vectorizer.transform([question]))
        xgb_prediction = label_encoder.inverse_transform(xgb_prediction_encoded)[0]

        # Find the most frequent prediction
        final_prediction = Counter([nb_prediction, svm_prediction, xgb_prediction]).most_common(1)[0][0]

        predictions.append(final_prediction)

    return predictions

def save_predictions_to_csv(questions, predictions):
    df = pd.DataFrame({'Questions': questions, 'Predictions': predictions})
    df.to_csv('questions.csv', index=False)
    return df

def get_csv(file_path):
    q=extract_questions_from_pdf(file_path)
    p=predict_questions(q)
    return save_predictions_to_csv(q,p)
    
def scorecheck(df):
    score = [0,0,0]
    for pred in df['Predictions']:
        if pred == 'Synthesis':
            score[0]=3
        if pred == 'Evaluation':
            score[1]=2
        if pred == 'Analyse':
            score[2]=1
    
    return sum(score)


