import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import pandas as pd
pd.read_csv()

def main():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = 2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    a=tokenizer(['how are you?','I am fine fine fine fine fine'],max_length=100,padding = True)
    print(a)
    b=model(a)
    print(b)

if __name__ == '__main__':
    main()