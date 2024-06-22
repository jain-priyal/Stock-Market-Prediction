import os
from collections import Counter
import sys
import pandas as pd
import torch
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification

'''
Function to load optimised BERT model.

If you do not have BERT.ckpt download it from here:
https://drive.google.com/drive/folders/1gD_xmFYmZAobbHgjkmRtDo37QILTKJAV
'''
def load_bert_model():
    # Load pre trained model
    model = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-yelp-polarity",
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    # Detect if GPU is available and set the device accordingly.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load model weights from a checkpoint if it exists, otherwise start with base model weights.
    if os.path.isfile('BERT.ckpt'):
        checkpoint = torch.load('BERT.ckpt', map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("Model loaded successfully from BERT.ckpt")
    else:
        print("No checkpoint found. Starting with base model weights.")
    return model, device

'''
Function which analyses the sentiment for a list of strings
'''
def analyze_sentiments(texts, model, tokenizer, device):
    model.eval()
    sentiments = []
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    # Process each text string in the input list.
    for text in texts:
        if isinstance(text, str) and text.strip():
            # Preprocess the text to match the model's expected input format.
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Disabling gradient calculations
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                sentiment = sentiment_labels[torch.argmax(probs).item()]
                sentiments.append(sentiment)
        else:
            # Handle non-string or empty inputs
            sentiments.append('Unknown')
    return sentiments

'''
Function to process sentiments for stock market summaries related to a specific company code.
'''
def process_stock_sentiments(file_path, company_code):
    df = pd.read_excel(file_path)
    tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
    model, device = load_bert_model()

    # Filter data for a specific company code
    filtered_df = df[(df['Company Code'] == company_code) & (df['Sentiment'] != 'NA') & pd.notna(df['Sentiment'])]
    if filtered_df.empty:
        return f"No data available for company code {company_code}"

    # Make sure to work only with rows that have non-null summaries
    valid_rows = filtered_df.dropna(subset=['Summary'])
    texts = valid_rows['Summary'].tolist()
    sentiments = analyze_sentiments(texts, model, tokenizer, device)
    
    # Assign sentiments back to valid_rows DataFrame
    valid_rows['Sentiment'] = sentiments

    # Analyze general sentiment
    sentiment_counts = Counter(sentiments)
    general_sentiment = max(sentiment_counts, key=sentiment_counts.get, default="No clear sentiment")

    # Finding the most positive and most negative URL
    most_positive_url = valid_rows[valid_rows['Sentiment'] == 'Positive']['URL'].iloc[0] if 'Positive' in sentiments else "No positive sentiments found."
    most_negative_url = valid_rows[valid_rows['Sentiment'] == 'Negative']['URL'].iloc[0] if 'Negative' in sentiments else "No negative sentiments found."

    return {
        'Most Positive URL': most_positive_url,
        'Most Negative URL': most_negative_url,
        'General Sentiment': general_sentiment
    }

'''
Wrapper function for output in appropriate format for gradio
'''
def get_stock_sentiments(company_code):
    results = process_stock_sentiments('cleaned-dataset-demo.xlsx', company_code)
    return results['General Sentiment'], results['Most Positive URL'], results['Most Negative URL']

'''
Main function
'''
if __name__ == "__main__":
    interface = gr.Interface(
        fn=get_stock_sentiments,  # the function to call
        inputs=gr.Textbox(label="Enter A Company Code (E.g. QAN)", placeholder="Type a company code here..."),
        outputs=[
            gr.Textbox(label="General Sentiment"),            # Label for the first output
            gr.Textbox(label="Most Confident Sentiment"),  # Label for the second output
            gr.Textbox(label="Most Bearish Sentiment")   # Label for the third output
        ],
        examples=['CBA', 'QAN', 'CSL', 'QBE', 'WTC'],  # Example inputs to show in the interface
        description="Enter a company code from ASX200 companies to get sentiment analysis results from the past 30 days.",
        title="Stock Sentiment Analysis",
        theme="huggingface"
    )

    interface.launch()


