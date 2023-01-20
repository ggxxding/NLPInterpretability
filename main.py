import torch
from transformers import BertModel, BertConfig, BertTokenizer

def main():
    model = BertModel.from_pretrained('bert-base')



if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    a=tokenizer(['how are you?','I am fine fine fine fine fine'],max_length=100,padding = True)
    print(a)