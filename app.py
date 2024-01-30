# app.py
from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load('filename.h5'))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=1).item()
        sentiment = sentiment_labels[predictions]

        return render_template('outputPage.html', text=text, sentiment=sentiment)
    
    except Exception as e:
        print('Error:', e)
        return 'An error occurred during prediction.'

if __name__ == '__main__':
    app.run(port=5000, debug=True)
