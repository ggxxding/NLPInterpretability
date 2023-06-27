import torch
import torch.nn as nn
from transformers import BertForSequenceClassification ,BertTokenizer, AdamW
from data_util import dataset
from jsondatautil import JsonDataset
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--do_train",
                    default = False,
                    action = "store_true",
                    help = 'default: False')
parser.add_argument("--do_eval",
                    default = False,
                    action = "store_true",
                    help = 'default: False')

parser.add_argument("--dataset",
                    default = 'snli_ori',
                    type = str,
                    help = 'snli_ori(default)/snli_aug/anli/hans')
parser.add_argument("--model",
                    default = 'bert-base-uncased',
                    type = str,
                    help = 'model name, default: bert-base-uncased')

args = parser.parse_args()
model_name = 'bert-base-uncased'

# hyperparam ori 75.9acc  0.3hidden 25epoch 32batch
hidden_dropout_prob = 0.3
num_labels = 3
epochs =20
batch_size = 16 #64
learning_rate = 1e-5
weight_decay = 1e-2

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")



def main():
    if not args.do_train and not args.do_eval:
        raise ValueError("At lease one of 'do_train' or 'do_eval' must be True")
    if args.do_train:
        if args.dataset == 'snli_ori':
            train_dir = '../data/NLI/original/train.tsv'
            # dev_dir = '../data/NLI/original/dev.tsv'
            dev_dir = '../data/snli/snli_1.0_test.txt' #test

        elif args.dataset == 'snli_aug':
            train_dir = '../data/NLI/all_combined/train.tsv'
            # dev_dir = '../data/NLI/all_combined/dev.tsv'
            dev_dir = '../data/NLI/all_combined/test.tsv' #test
        elif args.dataset == 'anli':
            train_dir = '../data/anli_v1.0/R1/train.jsonl'
            dev_dir = '../data/anli_v1.0/R1/dev.jsonl'

        if args.dataset == 'anli':
            train_dataset = JsonDataset(train_dir)
            valid_dataset = JsonDataset(dev_dir)
        else:
            train_dataset = dataset(train_dir)
            valid_dataset = dataset(dev_dir)

        train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size ,shuffle=True)
        valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=batch_size ,shuffle=False)

        model = BertForSequenceClassification.from_pretrained(args.model, 
                                                              num_labels = num_labels, 
                                                              hidden_dropout_prob = hidden_dropout_prob,
                                                              output_hidden_states = True)
        model.to(device)
        tokenizer = BertTokenizer.from_pretrained(args.model)

        # Prepare optimizer and schedule (linear warmup and decay)
        # 设置 bias 和 LayerNorm.weight 不使用 weight_decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # no_decay = ['bias', 'gamma', 'beta']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in list(model.named_parameters()) if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        # {'params': [p for n, p in list(model.named_parameters()) if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        # ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for i in range(epochs):
            train_loss, train_acc = train(model, tokenizer, train_dataloader, optimizer, criterion, device)
            print('epoch %2d, train loss: %.3f, train_acc: %.3f'%(i,train_loss,train_acc))
            valid_loss, valid_acc = evaluate(model, valid_dataloader, tokenizer, device)
            print('        , valid loss: %.3f, valid_acc: %.3f'%(valid_loss,valid_acc))
    if args.do_eval:
        if args.dataset == 'snli_ori':
            eval_dir = '../data/snli/snli_1.0_test.txt'
        elif args.dataset == 'snli_aug':
            eval_dir = '../data/NLI/all_combined/test.tsv'
        elif args.dataset == 'anli':
            eval_dir = '../data/anli/R1/test.jsonl'

        if args.dataset == 'anli':
            test_dataset = JsonDataset(eval_dir)
        else:
            test_dataset = dataset(eval_dir)
        test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size ,shuffle=False)

        # output[2] = hidden states, [L1,...,L12,output]
        model = BertForSequenceClassification.from_pretrained(args.model, 
                                                              num_labels = num_labels, 
                                                              hidden_dropout_prob = hidden_dropout_prob,
                                                              output_hidden_states = True)
        model.load_state_dict(torch.load( './data/' + args.model + '_'+ args.dataset + '.pt' ) )
        print('model loaded from %s'%('./data/' + args.model + '_'+ args.dataset + '.pt'))
        model.to(device)
        tokenizer = BertTokenizer.from_pretrained(args.model)
        test_loss, test_acc = test(model, test_dataloader, tokenizer, device)
        print('test acc: %.3f'% test_acc)



def train(model, tokenizer, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss  = 0
    epoch_acc   = 0
    for i,batch in enumerate(dataloader):
        text   = batch[0]
        text = list(zip( text[0], text[1] ))
        label = torch.tensor(batch[1]).to(device)

        inp = tokenizer(text, padding = 'max_length', truncation = True, max_length = 128, return_tensors = 'pt').to(device)
        optimizer.zero_grad()
        output = model(**inp, labels = label)

        
        pred_prob = output[1]
        pred_label = pred_prob.argmax(dim = 1)
        # loss = criterion(pred_prob.view(-1,num_labels), label.view(-1))
        loss = output[0]

        acc = ((pred_label == label.view(-1)).sum()).item()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

        if i % 100 == 0 :
            print("step %3d, loss: %.3f, acc: %.3f"%( i+1, epoch_loss/(i+1), epoch_acc/((i+1)*len(label))))
    torch.save(model.state_dict(),  './data/' + args.model +'_'+ args.dataset + '.pt')
    print('model saved at: %s'%( './data/' + args.model + '_'+ args.dataset + '.pt' ))
    return epoch_loss/len(dataloader), epoch_acc/len(dataloader.dataset)


def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            text   = batch[0]
            text = list(zip( text[0], text[1] ))
            label = torch.tensor(batch[1]).to(device)

            inp = tokenizer(text, padding = 'max_length', truncation = True, max_length = 128, return_tensors = 'pt').to(device)

            output = model(**inp, labels=label)

            pred_label = output[1].argmax(dim=1)
            loss = output[0]
            acc = ((pred_label == label.view(-1)).sum()).item()

            epoch_loss += loss.item()
            epoch_acc += acc

    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader.dataset)

def test(model, dataloader, tokenizer, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            text = batch[0]
            text = list(zip( text[0], text[1] ))
            label = torch.tensor(batch[1]).to(device)
            inp = tokenizer(text, padding = 'max_length', truncation = True, max_length = 128, return_tensors = 'pt').to(device)
            output = model(**inp, labels=label)
            # print(output[2][12][:,0])# [CLS] [16(batch),768]

            pred_label = output[1].argmax(dim=1)
            loss = output[0]
            acc = ((pred_label == label.view(-1)).sum()).item()

            # print(output[2][12][:,0])  #[batchsize, 128, 768]
            # print(output[2][12][:,0].shape) #[batchsize,768] 即CLS标签


            test_loss += loss.item()
            test_acc += acc
    return test_loss / len(dataloader), test_acc/ len(dataloader.dataset)


if __name__ == '__main__':
    main()

    # for epoch in range(1):
    #     for i,data in enumerate(dataloader,0):
    #         print(i,data)
    #         break