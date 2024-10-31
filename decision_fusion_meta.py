import os
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import csv
import openpyxl
from torch.xpu import device

# from pretty_confusion_matrix import pp_matrix
import time
import torch
import torch.nn as nn
from torchvision import transforms

from lstm_model import Lstm
from resnet18_model import resnet18
from decision_fusion_ave import PicData
from torch.utils.data import Dataset, DataLoader

data_train_path = os.path.abspath('D:/毕设工作/new_train.mat')
data_test_path = os.path.abspath('D:/毕设工作/new_test.mat')

# 读取mat文件中的阻抗数据以及标签
data_train_all = scio.loadmat(data_train_path)
data_train = data_train_all['all_cell_train']  # 5578*1 cell
label_train = data_train_all['all_label_train']

data_test_all = scio.loadmat(data_test_path)
data_test = data_test_all['all_cell_test']  # 5578*1 cell
label_test1 = data_test_all['all_label_test']

# 加载预训练的模型权重
model1 = Lstm()
model2 = resnet18()

weight_path1 = './99.1Lstm.pth'
weight_path2 = './resnet18-final.pth'

mean = torch.tensor([1.3106e+05, 1.3036e+05, 1.3027e+05, 1.2984e+05, -0.0029, -0.0292, -0.0369, -0.0561])
std = torch.tensor([6.6278e+03, 6.604e+03, 6.5823e+03, 6.3055e+03, 0.0075, 0.0786, 0.1019, 0.1544])

model1.load_state_dict(torch.load(weight_path1,weights_only=True))
in_channel = model2.fc.in_features
model2.fc = nn.Linear(in_channel, 4)
model2.load_state_dict(torch.load(weight_path2, map_location='cpu',weights_only=True))


class MetaLearn(nn.Module):
    def __init__(self, input_dim=8, output_dim=4):
        super(MetaLearn, self).__init__()
        self.input = input_dim
        self.output = output_dim
        self.fc1 = nn.Linear(self.input, 128)
        self.fc2 = nn.Linear(128, self.output)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


def record_result(pred_list, label_list):
    file_path = 'mate_pred.xlsx'

    mylist = [pred_list, label_list]

    column = ['pred_list', 'label_list']
    test = pd.DataFrame(mylist)
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        test.to_excel(writer, sheet_name='Sheet1', index=False)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.037, 0.037, 0.037], [0.0105, 0.0105, 0.0105])
                                        ])
    img_transform1 = transforms.Compose([transforms.Resize((224, 224)),

                                         transforms.ToTensor(),
                                         transforms.Normalize([0.037, 0.037, 0.037], [0.0105, 0.0105, 0.0105])
                                         ])

    image_train_path = os.path.abspath('D:/毕设工作/input_train')
    image_test_path = os.path.abspath('D:/毕设工作/input_test')

    # 加载数据集
    all_data = PicData(image_train_path, transform=img_transform, data_from=data_train, label_from=label_train)
    all_dest = PicData(image_test_path, transform=img_transform1, data_from=data_test, label_from=label_test1)
    all_data_train, all_data_val = torch.utils.data.random_split(all_data, [round(0.83 * len(all_data)),
                                                           round(0.17 * len(all_data))])
    train_num = len(all_data_train)
    val_num = len(all_data_val)
    test_num = len(all_dest)
    #
    input_train, y_train = get_decision(all_data_train)
    input_test, y_test = get_decision(all_data_val)
    input_val, y_val = get_decision(all_dest)

    batch_size = 64
    train_loader = DataLoader(dataset(input_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset(input_test, y_test), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset(input_val, y_val), batch_size, shuffle=False)

    net = MetaLearn()
    # net.to(device)
    loss_function = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    epochs = 20
    best_acc = 0.0
    save_path = './{}.pth'.format('MetaLearn')
    train_steps = len(train_loader)
    test_steps = len(test_loader)
    trloss = []
    tracc = []
    tloss = []
    tacc = []


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for epoch in range(epochs):
        # train
        net.train()
        acct = 0.0
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            decision, labels = data
            optimizer.zero_grad()
            decision = torch.Tensor(decision)
            decision = decision.to(device)
            labels = labels.to(device)
            outputs = net(decision)
            predict_x = torch.max(outputs, dim=1)[1]
            acct += torch.eq(predict_x, labels).sum().item()

            loss = loss_function(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_acc = acct / train_num
        tracc.append(train_acc)
        trloss.append(running_loss/train_steps)

        # test
        net.eval()
        acc = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                decision, label_test = test_data

                decision = torch.Tensor(decision)
                decision = decision.to(device)
                label_test = label_test.to(device)
                outputs = net(decision)

                loss = loss_function(outputs, label_test.long())
                test_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, label_test).sum().item()

        scheduler.step()

        test_accurate = acc / val_num
        tloss.append(test_loss/test_steps)
        tacc.append(test_accurate)
        print('[epoch %d] train_ave_loss: %.3f  train_accuracy: %.3f  test_ave_loss: %.3f test_accuracy: '
              '%.3f learning rate: %.8f' %
              (epoch + 1, running_loss / train_steps, train_acc, test_loss/test_steps, test_accurate, lr))

        if test_accurate > best_acc:
            best_acc = test_accurate
            torch.save(net.state_dict(), save_path)

    print(best_acc)
    print('Finished Training')

    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(start.elapsed_time(end))

    #
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # conf_matrix = torch.zeros(4, 4)
    # weight_path = './MetaLearn.pth'
    # net.load_state_dict(torch.load(weight_path))
    # pred_list = []
    # label_list = []
    #
    # net.eval()
    # with torch.no_grad():
    #     for data in val_loader:
    #         fea, label = data
    #         label = label.squeeze()
    #         fea = torch.Tensor(fea)
    #         out = net(fea)
            # conf_matrix, preds, labels = confusion_matrix(out, label, conf_matrix)
            # pred_list.append(preds.flatten())
            # label_list.append(labels.flatten())
    # conf_matrix = np.array(conf_matrix)
    # df_cm = pd.DataFrame(conf_matrix, index=['Eos', 'Lym', 'Mono', 'Neu'], columns=['Eos', 'Lym', 'Mono', 'Neu'])
    # pp_matrix(df_cm)
    # # plot_matrix(conf_matrix, [0, 1, 2, 3], title='confusion_matrix_svc',
    # #             axis_labels=['Eos', 'Lym', 'Mono', 'Neu'])
    # x1 = range(0, epochs)
    # y1 = tracc
    # y2 = trloss
    # y3 = tacc
    # y4 = tloss
    #
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(x1, y2, 'ro-', label="train loss")
    # plt.plot(x1, y4, 'bs-', label="test loss")
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(x1, y1, 'ro-', label="train acc")
    # plt.plot(x1, y3, 'bs-', label="test acc")
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("acc")
    # plt.legend()
    # plt.show()
    # plt.savefig('./Meta_result')
    #
    # # record_result(pred_list, label_list)


class dataset(Dataset):
    def __init__(self, decision, label):
        self.len = len(decision)
        self.decisions = decision
        self.label = torch.Tensor(label)

    def __getitem__(self, idx):
        return self.decisions[idx], self.label[idx]

    def __len__(self):
        return self.len


def get_decision(data):
    img_list = [x[0] for x in data]
    data_list = [x[1] for x in data]
    label = [x[2] for x in data]
    num = len(data)

    model1.eval()
    with torch.no_grad():
        for idx in range(0, num):
            data = torch.unsqueeze(data_list[idx], dim=0)
            output, _ = model1(data)
            data_list[idx] = torch.squeeze(output, dim=0)

    model2.eval()
    with torch.no_grad():
        for idx in range(0, num):
            img = torch.unsqueeze(img_list[idx], dim=0)
            output, _ = model2(img)
            img_list[idx] = torch.squeeze(output, dim=0)

    decision = []
    for idx in range(0, num):
        a = data_list[idx]
        b = img_list[idx]
        x = torch.cat((a, b), dim=0)
        decision.append(x)
    return decision, label


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.max(preds, 1)[1].unsqueeze(dim=-1).numpy()
    preds = preds.astype(int)
    labels = labels.unsqueeze(dim=-1).numpy()
    labels = labels.astype(int)
    for idx in range(int(len(preds))):
        conf_matrix[labels[idx, 0], preds[idx, 0]] += 1
    return conf_matrix, preds, labels


if __name__ == '__main__':
    main()

