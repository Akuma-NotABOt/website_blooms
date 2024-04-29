import pandas as pd
from nltk import word_tokenize, pos_tag
from collections import Counter
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_action_verbs(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    action_verbs = [word for word, pos in tagged_tokens if pos.startswith('VB') and len(word) >= 3]
    return action_verbs

def process_class_questions(df,class_name):
    class_questions = df[df['Predictions'] == class_name]['Questions']
    action_verbs = []
    for question in class_questions:
        action_verbs += extract_action_verbs(question)
    return action_verbs

def sig_verbs(df):
    classes = df['Predictions'].unique()
    significant_verbs_by_class = {}
    for class_name in classes:
        action_verbs = process_class_questions(df,class_name)
        verb_counter = Counter(action_verbs)
        significant_verbs = [(verb, count) for verb, count in verb_counter.items() if len(verb) >= 3]
        significant_verbs_by_class[class_name] = sorted(significant_verbs, key=lambda x: x[1], reverse=True)[:5] 
    return significant_verbs_by_class
