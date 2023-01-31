import torch
import torch.nn as nn
from transformers import BertForSequenceClassification ,BertTokenizer, AdamW
from data_util import dataset
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
do_train = True

# hyperparam
hidden_dropout_prob = 0.3
num_labels = 3
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 3
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")



def main():
    if do_train:
        train_dataset = dataset('/Users/ggxxding/Documents/GitHub/data/snli/snli_1.0_train.txt')
        train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size ,shuffle=True)

        valid_dataset = dataset('/Users/ggxxding/Documents/GitHub/data/snli/snli_1.0_dev.txt')
        valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=batch_size ,shuffle=True)

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = num_labels, hidden_dropout_prob = hidden_dropout_prob)
        model.to(device)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inp=tokenizer(('test a test','test test'), padding = 'max_length', max_length=6, add_special_tokens =True,return_tensors='pt').to('mps')
        out= model(**inp,labels = torch.tensor([0,1]).to('mps'))

        # 定义优化器和损失函数
        # Prepare optimizer and schedule (linear warmup and decay)
        # 设置 bias 和 LayerNorm.weight 不使用 weight_decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for i in range(epochs):
            train_loss, train_acc = train(model, tokenizer, train_dataloader, optimizer, criterion, device)
            print('train_loss: ', train_loss, '\t train_acc: ', train_acc)
            valid_loss, valid_acc = evaluate(model, valid_dataloader, tokenizer, device)
            print('valid_loss: ', valid_loss, '\t valid_acc: ', valid_acc)



def train(model, tokenizer, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss  = 0
    epoch_acc   = 0
    for i,batch in enumerate(dataloader):
        text   = batch[0]
        label = torch.tensor(batch[1]).to(device)

        inp = tokenizer(text, padding = 'max_length', truncation = True, max_length = 128, return_tensors = 'pt').to(device)
        optimizer.zero_grad()
        output = model(**inp, labels = label)
        
        pred_prob = output.logits
        pred_label = pred_prob.argmax(dim = 1)
        # loss = criterion(pred_prob.view(-1,num_labels), label.view(-1))
        loss = output.loss

        acc = ((pred_label == label.view(-1)).sum()).item()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

        if i%10 == 0 :
            print("step ",i+1 ,"current loss:", epoch_loss/(i+1), "current acc:", epoch_acc/((i+1)*len(label)))
    return epoch_loss/len(dataloader), epoch_acc/len(dataloader.dataset)


def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            text   = batch[0]
            label = torch.tensor(batch[1]).to(device)

            inp = tokenizer(text, padding = 'max_length', truncation = True, max_length = 128, return_tensors = 'pt').to(device)

            output = model(**inp, labels=label)

            pred_label = output.logits.argmax(dim=1)
            loss = output.loss
            acc = ((pred_label == label.view(-1)).sum()).item()

            epoch_loss += loss.item()
            epoch_acc += acc

    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader.dataset)


if __name__ == '__main__':
    main()

    # for epoch in range(1):
    #     for i,data in enumerate(dataloader,0):
    #         print(i,data)
    #         break