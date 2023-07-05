import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

class dataset(Dataset):
    def __init__(self,filepath):
        data = pd.read_csv(filepath, sep='\t')
        data = data.dropna(subset = ['sentence1','sentence2','gold_label'],how = 'any') #Train: 550152 -> 550146
        data = data[  -data.gold_label.isin(['-'])  ] # Train : 550146 -> 549361 (785 '-' labels)
        self.sentence1 = data.sentence1.values
        self.sentence2 = data.sentence2.values
        self.label = data.gold_label.values
        self.len = data.shape[0]
        self.label_dict = { 
            'entailment':2,
            'neutral':1,
            'contradiction':0}
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return self.len
    def __getitem__(self, item):
        tokens = [self.sentence1[item],  self.sentence2[item]]
        # ids = self.tokenizer.encode(tokens, padding = 'max_length', max_length = 128)
        return tokens, self.label_dict[self.label[item]]
    
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
    
class HansDataset(Dataset):
    def __init__ (self,filepath):
        data = pd.read_csv(filepath, sep='\t')
        data = data.dropna(subset = ['sentence1','sentence2','gold_label'],how = 'any')
        data = data[  -data.gold_label.isin(['-'])  ]
        self.sentence1 = data.sentence1.values
        self.sentence2 = data.sentence2.values
        self.label = data.gold_label.values
        self.len = data.shape[0]
        self.label_dict = { 
            'entailment':2,
            'non-entailment':1}
    def __len__(self):
        return self.len
    def __getitem__(self,item):
        tokens = [self.sentence1[item],  self.sentence2[item]]
        return tokens, self.label_dict[self.label[item]]

if __name__ == '__main__':
    dataset1 = dataset('/Users/ggxxding/Documents/GitHub/data/snli/snli_1.0_train.txt')
    dataloader = DataLoader(dataset=dataset1,batch_size=32,shuffle=False)
    for epoch in range(1):
        for i,data in enumerate(dataloader,0):
            print(i,data)
            print(len(data[0][0]))
            break