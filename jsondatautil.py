import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertTokenizer

class JsonDataset(Dataset):
    def __init__(self, filepath):
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        self.data = pd.DataFrame(data)
        # convert the list of dictionaries to pandas dataframe for easier manipulation
        data = pd.DataFrame(data)
        self.context = data.context.values
        self.hypothesis = data.hypothesis.values
        self.label = data.label.values
        self.len = data.shape[0]
        self.label_dict = { 
            'e': 2,
            'n': 1,
            'c': 0}
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return self.len
    def __getitem__(self, item):
        tokens = [self.context[item],  self.hypothesis[item]]

        # ids = self.tokenizer.encode(tokens, padding = 'max_length', max_length = 128)
        return tokens, self.label_dict[self.label[item]]

if __name__ == '__main__':
    dataset1 = JsonDataset(r'../data/anli_v1.0/R3/train.jsonl')
    dataloader = DataLoader(dataset=dataset1, batch_size=32, shuffle=False)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max = 0
    for epoch in range(1):
        for i, data in enumerate(dataloader, 0):
            text   = data[0]
            text = list(zip( text[0], text[1] ))
            for i in text:
                t = i[0] +' '+i[1]
                l = len(t.split())
                if l>max:
                    max = l
                    print(t.split())
                    inp = tokenizer(i, padding = 'max_length', truncation = True, max_length = 256, return_tensors = 'pt')
                    print(inp)
    print(max)
