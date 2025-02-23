import torch
import torch.nn as nn
import copy
import random
import warnings
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score,precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold, RepeatedKFold
warnings.filterwarnings('ignore')



def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed(777)

device = torch.device("cuda:0")


train_data_path = pd.read_csv("cdhit/0.7/train_set7_label.csv")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
def tokenize(path):
    data_frame = path
    data_columns = data_frame.columns.tolist()
    data_columns = [i for i in data_columns]
    data_frame.columns = data_columns
    trainlabel = data_frame[data_frame.columns[0]]
    proBert_seq = data_frame[data_frame.columns[1]]
    return  np.array(trainlabel), np.array(proBert_seq)

train_label, train_X = tokenize(train_data_path)




def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    Pre = TP / (TP + FP)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    # 计算BACC
    BACC = (TPR + TNR) / 2
    return SN, SP, ACC, MCC,BACC, Pre


def cal_score(pred, label):
    try:
        AUC = roc_auc_score(list(label), pred)

    except:
        AUC = 0
    # pred = np.around(pred)
    pred = np.argmax(pred, axis=1)
    # pred = pred.argmax(dim=1)
    # print(pred)
    label = np.array(label)
    # print(label)
    cm = confusion_matrix(label, pred, labels=None, sample_weight=None)
    SN, SP, ACC, MCC,BACC, Pre = Model_Evaluate(cm )
    print(
        "Model score --- SN:{0:.3f}       SP:{1:.3f}       ACC:{2:.3f}       MCC:{3:.3f}   BACC:{4:.3f}    Pre:{4:.3f}   AUC:{5:.3f}".format(
            SN, SP, ACC, MCC,BACC,Pre, AUC))

    return BACC,cm


class AQYDataset(Dataset):
    def __init__(self, df, label, device):
        self.protein_seq = df
        self.label_list = label

    def __getitem__(self, index):
        seq = self.protein_seq[index]
        seq = seq.replace('', ' ')
        encoding = tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            max_length=50,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            # 'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),

        }
        label = self.label_list[index]
        return sample,  label

    def __len__(self):
        return len(self.protein_seq)


def fit(model, train_loader, optimizer,  device):
    model.train()
    outlist = []
    pred_list = []
    label_list = []

    accumulation_steps = 2
    for i, (samples,  label) in enumerate(train_loader):
    # for seq1, seq2, label, label1, label2,  seq, label_sum in train_iter_cont:
    #     input_ids = seq['input_ids'].to(device)
    #     input_ids1 = seq1['input_ids'].to(device)
    #     input_ids2 = seq2['input_ids'].to(device)
    #     attention_mask = seq['attention_mask'].to(device)
    #     attention_mask1 = seq1['attention_mask'].to(device)
    #     attention_mask2 = seq2['attention_mask'].to(device)
        input_ids = samples['input_ids'].to(device)
        # token_type_ids = samples['token_type_ids'].to(device)
        attention_mask = samples['attention_mask'].to(device)
        label = torch.tensor(label).to(device)
        pred = model(input_ids, attention_mask)
        # print(label)
        # print("预测",pred)
        # pred1 = model(input_ids1, attention_mask1)
        # pred2 = model(input_ids2, attention_mask2)
        pred = pred.squeeze()
        # print(pred)
        # print(label)
        # pred1 = pred1.squeeze()
        # pred2 = pred2.squeeze()
        # loss1 = ContrastiveLoss_cov(pred1, pred2, label)
        loss2 = criterion(pred, label)
        loss = loss2
        # loss = FocalLoss(alpha=4,gamma=3)(pred,label)
        loss.backward()
        # if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        # pred_list.extend(pred.squeeze().cpu().detach().numpy())
        pred_list.extend(pred.squeeze().cpu().detach().numpy())

        label_list.extend(label.squeeze().cpu().detach().numpy())
    pred_list = np.array(pred_list)
    label_list = np.array(label_list)
    score,_ = cal_score(pred_list, label_list)
    return score,outlist


def validate(model, val_loader, device):
    model.eval()
    pred_list = []
    label_list = []

    for samples,  label in val_loader:
        input_ids = samples['input_ids'].to(device)
        # token_type_ids = samples['token_type_ids'].to(device)
        attention_mask = samples['attention_mask'].to(device)
        label = torch.tensor(label).to(device)
        pred= model(input_ids, attention_mask)
        # pred_list.extend(pred.squeeze().cpu().detach().numpy())
        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())
    pred_list = np.array(pred_list)
    label_list = np.array(label_list)
    score,cm = cal_score(pred_list, label_list)
    # with open("../../twoproject/lstm_single/data/pre.csv", "a", encoding="utf-8", newline="") as f:
    #                 csv_writer = csv.writer(f)
    #                 csv_writer.writerow(pred_list)

    return score, pred_list, label_list,cm


# tokenizer = BertTokenizer.from_pretrained('/home/hd/SGao/probert')
tokenizer = AutoTokenizer.from_pretrained("/home/hd/SGao/esm2_t30")



class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # print(x.shape)
        query = self.query(x)
        # print(query.shape)
        key = self.key(x)
        # print(key.shape)
        value = self.value(x)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention = self.softmax(scores)
        output = torch.matmul(attention, value)
        return output
#lstm
# class CNN_BiLSTM_Attention(nn.Module):
#
#     def __init__(self, embedding_dim=50, hidden_dim=32, n_layers=1):
#         super(CNN_BiLSTM_Attention, self).__init__()
#         self.bert = AutoModel.from_pretrained("/home/hd/SGao/new/esm2")
#         # self.bert = BertModel.from_pretrained("/home/hd/SGao/new/protbert")
#         out_channle = 32
#         self.lstm1 = nn.LSTM(1280,512,num_layers=1,bidirectional = True,batch_first = True)
#         self.lstm2 = nn.LSTM(1024, 128, num_layers=1, bidirectional=True, batch_first=True)
#
#         self.fc1 = nn.Linear(256, 16)
#         self.fc2 = nn.Linear(16, 1)
#         self.self_attention = SelfAttention(out_channle)
#
#         self.relu = nn.ReLU()
#         self.lrelu = nn.LeakyReLU()
#
#         self.dropout = nn.Dropout(0.2)
#
#         # self.w_omega = nn.Parameter(torch.Tensor(out_channle, out_channle))
#         # self.u_omega = nn.Parameter(torch.Tensor(out_channle, 1))
#         #
#         # nn.init.uniform_(self.w_omega, -0.1, 0.1)
#         # nn.init.uniform_(self.u_omega, -0.1, 0.1)
#
#     def attention_net(self, x):
#         u = torch.tanh(torch.matmul(x, self.w_omega))
#         att = torch.matmul(u, self.u_omega)
#         att_score = F.softmax(att, dim=1)
#         scored_x = x * att_score
#         context = torch.sum(scored_x, dim=1)
#         return context
#
#     def forward(self, input_ids, attention_mask):
#         pooled_output, hidden_states = self.bert(
#             input_ids=input_ids,
#             # token_type_ids=token_type_ids,
#             attention_mask=attention_mask,
#             return_dict=False
#         )
#         # [32, 50, 1280])
#
#         pooled_output,_ = self.lstm1(pooled_output.float())
#         # print(pooled_output.shape)
#         pooled_output = self.dropout(pooled_output)
#         pooled_output,_ =self.lstm2(pooled_output)
#         pooled_output = self.dropout(pooled_output)
#         bi_lstm_out = torch.mean(pooled_output,axis=1,keepdim=True)
#         # attn_output = self.self_attention(bi_lstm_out)
#
#         out2 = self.fc1(bi_lstm_out)
#         logit = self.fc2(out2)
#
#         return nn.Sigmoid()(logit)
#

#全部模型
# class CNN_BiLSTM_Attention(nn.Module):
#     def __init__(self, embedding_dim=50, hidden_dim=32, n_layers=1):
#         super(CNN_BiLSTM_Attention, self).__init__()
#         self.bert = AutoModel.from_pretrained("/home/hd/SGao/new/esm2")
#         for param in self.bert.parameters():
#             param.requires_grad = False
#         out_channle = 32
#
#         # self.conv1 = nn.Conv1d(1280, 512, kernel_size=7, stride=1, padding='same')
#         # self.conv2 = nn.Conv1d(512, 128, kernel_size=7, stride=1, padding='same')
#         # self.conv3 = nn.Conv1d(128, 32, kernel_size=7, stride=1, padding='same')
#
#         self.conv1 = nn.Conv1d(1280, 512, kernel_size=5, stride=1, padding='same')
#         self.conv2 = nn.Conv1d(512, 128, kernel_size=5, stride=1, padding='same')
#         # self.conv3 = nn.Conv1d(256, 128, kernel_size=5, stride=1, padding='same')
#         self.conv3 = nn.Conv1d(128, 32, kernel_size=5, stride=1, padding='same')
#
#         self.batch1 = nn.BatchNorm1d(512)
#         # self.batch2 = nn.BatchNorm1d(256)
#         self.batch2 = nn.BatchNorm1d(128)
#         self.batch3 = nn.BatchNorm1d(32)
#
#         self.fc1 = nn.Linear(32, 16)
#         self.fc2 = nn.Linear(16, 1)
#
#         self.self_attention = SelfAttention(out_channle)
#
#         self.relu = nn.ReLU()
#         self.lrelu = nn.LeakyReLU()
#
#         self.dropout = nn.Dropout(0.2)
#         self.dropout1 = nn.Dropout(0.5)
#     def forward(self, input_ids, attention_mask):
#         # print(self.bert)
#         pooled_output,hidden_states = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=False
#         )
#         # [32, 50, 1280])
#
#         pooled_output = self.dropout(pooled_output)
#         imput = pooled_output.permute(0, 2, 1)
#
#         conv1_output = self.conv1(imput)
#         batch1_output = self.batch1(conv1_output)
#
#         conv2_output = self.conv2(batch1_output)
#         batch2_output = self.batch2(conv2_output)
#
#         conv3_output = self.conv3(batch2_output)
#         batch3_output = self.batch3(conv3_output)
#
#         # [32, 256,50, )
#         prot_out = torch.mean(batch3_output, axis=2, keepdim=True)
#         prot_out = prot_out.permute(0, 2, 1)
#         # prot_out = prot_out.squeeze(1)
#         attn_output = self.self_attention(prot_out)
#
#         out2 = self.fc1(attn_output )
#         # out2 = self.dropout1(out2)
#         logit = self.fc2(out2)
#
#         return nn.Sigmoid()(logit)

#只是使用ems2
class newmodel(nn.Module):

    def __init__(self, embedding_dim=50, hidden_dim=32, n_layers=1):
        super(newmodel, self).__init__()
        self.bert = AutoModel.from_pretrained("/home/hd/SGao/esm2_t30")
        # self.bert = BertModel.from_pretrained("/home/hd/SGao/probert")
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        out_channle = 32
        self.LSTM = nn.LSTM(640, 320, num_layers=2,
                          bidirectional=True, dropout=0.3)
        self.global_avg_pooling_layer = nn.AdaptiveAvgPool1d(1)
        # self.conv1 = nn.Conv1d(1024, 256, kernel_size=5, stride=1, padding='same')
        # self.conv2 = nn.Conv1d(256, 128, kernel_size=5, stride=1, padding='same')
        # self.conv3 = nn.Conv1d(128, 32, kernel_size=5, stride=1, padding='same')
        self.conv1 = torch.nn.Conv1d(in_channels=1024,
                                     out_channels=256,
                                     kernel_size=2,
                                     stride=1
                                     )
        self.conv2 = torch.nn.Conv1d(in_channels=1024,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=1
                                     )
        self.conv3 = torch.nn.Conv1d(in_channels=1024,
                                     out_channels=256,
                                     kernel_size=4,
                                     stride=1
                                     )
        self.conv4 = torch.nn.Conv1d(in_channels=1024,
                                     out_channels=256,
                                     kernel_size=5,
                                     stride=1
                                     )

        self.fc1 = nn.Linear(640, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)


        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=5)
        self.dropout = nn.Dropout(0.3)

        # self.w_omega = nn.Parameter(torch.Tensor(1280, 256))
        # self.u_omega = nn.Parameter(torch.Tensor(256, 1))
        #
        # nn.init.uniform_(self.w_omega, -0.1, 0.1)
        # nn.init.uniform_(self.u_omega, -0.1, 0.1)

    # def attention_net(self, x):
    #     u = torch.tanh(torch.matmul(x, self.w_omega))
    #     att = torch.matmul(u, self.u_omega)
    #     att_score = F.softmax(att, dim=1)
    #     scored_x = x * att_score
    #     context = torch.sum(scored_x, dim=1)
    #     return context


    def forward(self, input_ids,attention_mask):
        pooled_output,hidden_states = self.bert(
            input_ids=input_ids,
            # token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        y,_= self.LSTM(hidden_states)
        # x = self.dropout(hidden_states)
        # y = y.permute(0, 2, 1)



        # output = torch.mean(x,axis=2,keepdim=True)
        # flatten_output = output .reshape(output.shape[0], -1)
        # print(flatten_output.shape)
        # print(flatten_output.shape)
        out2 = self.fc1(y)
        # # out2 = self.relu(out2)
        out3 = self.fc2(out2)
        out3 = self.relu(out3)
        # logit,_ = self.gru(pooled_output)
        # # print(logit.shape)
        # logit = self.dropout(logit)
        # logit = logit.reshape(logit.shape[0], -1)
        # logit = self.fc1(logit)
        # # logit = self.attention_net(logit)
        # logit = self.relu( logit )
        # logit = self.fc2(logit)
        # logit = self.relu(logit)
        logit = self.fc3(out3)
        # return nn.Sigmoid()(logit)
        return logit

    # def trainModel(self, input_ids, attention_mask):
    #     # with torch.no_grad():
    #     # print("kankan0刚进来",x.shape)
    #     output = self.forward(input_ids, attention_mask)
    #     # print("outputkankan1:", output.shape)
    #
    #     # 模型的前向传播方法
    #     output  = self.fc3(output)
    #     # print(output)
    #     return output
    # def forward(self, input_ids, attention_mask):
    #     pooled_output,hidden_states = self.bert(
    #         input_ids=input_ids,
    #         # token_type_ids=token_type_ids,
    #         attention_mask=attention_mask,
    #         return_dict=False
    #     )
    #     # [32, 50, 1280])
    #     pooled_output = self.dropout(pooled_output)
    #     # flatten_output = pooled_output.view(32,64000)
    #     flatten_output = torch.mean(pooled_output, axis=1, keepdim=True)
    #     flatten_output = flatten_output.squeeze(1)
    #
    #     out2 = self.fc1(flatten_output )
    #     out2 = self.relu(out2)
    #     out3 = self.fc2(out2)
    #     out3= self.relu(out3)
    #     logit = self.fc3(out3)
    #
    #     return nn.Sigmoid()(logit)



if __name__ == '__main__':

    X_train = train_X
    label_train = train_label
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    skf = KFold(n_splits=5, shuffle=True, random_state=666)
    valid_pred = []
    valid_label = []

    for index, (train_idx, val_idx) in enumerate(skf.split(X_train, label_train)):

        print('**' * 10, '训练-测试开始', '**' * 10)
        proBer_train_seq,proBer_valid_seq  =X_train[train_idx], X_train[val_idx]
        train_Y , valid_Y  =label_train[train_idx], label_train[val_idx]

        train_dataset = AQYDataset(proBer_train_seq, train_Y, device)

        valid_dataset = AQYDataset(proBer_valid_seq, valid_Y, device)
        #
        # train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=32,
        #                                               shuffle=True, collate_fn=collate, drop_last=True)

        train_loader = DataLoader(train_dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=4)

        valid_loader = DataLoader(valid_dataset,
                                  batch_size=128,
                                  shuffle=False,
                                  drop_last=True,
                                  num_workers=4)

        model = newmodel().to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

        criterion = FocalLoss()
        # criterion = nn.CrossEntropyLoss(reduction="sum")

        best_val_score = float('-inf')
        last_improve = 0
        best_model = None

        for epoch in range(40):
            train_score,outlist = fit(model, train_loader, optimizer,  device)
            val_score, _, _,_ = validate(model, valid_loader, device)

            if val_score > best_val_score:
                best_val_score = val_score
                best_model = copy.deepcopy(model)
                last_improve = epoch
                improve = '*'
            else:
                improve = ''

            print(f'Epoch: {epoch + 1} Train Score: {train_score}, Valid Score: {val_score}  ')

        model = best_model
        torch.save(best_model.state_dict(), './num_model{}_acc{}.pkl'.format((index+1 + 1),  best_val_score))
        print(f"=============e0nd!!!!================")
        print("train")
        train_score, _, _,_ = validate(model, train_loader, device)
        print("valid")
        valid_score, pred_list, label_list,cm= validate(model, valid_loader, device)
        valid_pred.extend(pred_list)
        valid_label.extend(label_list)
        print(cm)
print("*****************************************5-fold  valid**********************************************")
#
print("valid_score")
cross_valid_score,cm = cal_score(valid_pred, valid_label)
print(cm)