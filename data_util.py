import pandas as pd
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self,filepath):
        data = pd.read_csv(filepath, sep='\t')
        data = data.dropna(subset = ['sentence1','sentence2','gold_label'],how = 'any') #Train: 550152 -> 550146
        data = data[  -data.gold_label.isin(['-'])  ] # Train : 550146 -> 549361 (785 '-' labels)
        self.sentence1 = data.sentence1.values
        self.sentence2 = data.sentence2.values
        self.label = data.gold_label.values
        self.len = data.shape[0]
        self.label_dict = {'contradiction':0, 
                           'neutral':1,
                           'entailment':2}
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return self.len
    def __getitem__(self, item):
        tokens = self.sentence1[item]+ ' [SEP] ' + self.sentence2[item]
        # ids = self.tokenizer.encode(tokens, padding = 'max_length', max_length = 128)
        return tokens, self.label_dict[self.label[item]]
    

if __name__ == '__main__':
    dataset1 = dataset('/Users/ggxxding/Documents/GitHub/data/snli/snli_1.0_train.txt')
    dataloader = DataLoader(dataset=dataset1,batch_size=32,shuffle=False)
    for epoch in range(1):
        for i,data in enumerate(dataloader,0):
            print(i,data)
            break