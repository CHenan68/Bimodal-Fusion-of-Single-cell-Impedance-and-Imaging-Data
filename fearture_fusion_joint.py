import os
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import openpyxl
# from pretty_confusion_matrix import pp_matrix_from_data, pp_matrix
import time

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from feature_lstm_model import Lstm
from feature_resnet18_model import resnet18
from featrue_fusion_marginal import PicData
from torch.utils.data import Dataset, DataLoader


data_train_path = os.path.abspath('D:/毕设工作/new_train.mat')
data_test_path = os.path.abspath('D:/毕设工作/new_test.mat')

# 读取mat文件中的阻抗数据以及标签
data_train_all = scio.loadmat(os.path.join(data_train_path))
data_train = data_train_all['all_cell_train']  # 5578*1 cell
label_train = data_train_all['all_label_train']

data_test_all = scio.loadmat(data_test_path)
data_test = data_test_all['all_cell_test']  # 985*1 cell
label_test1 = data_test_all['all_label_test']

mean = torch.tensor([1.3106e+05, 1.3036e+05, 1.3027e+05, 1.2984e+05, -0.0029, -0.0292, -0.0369, -0.0561])
std = torch.tensor([6.6278e+03, 6.604e+03, 6.5823e+03, 6.3055e+03, 0.0075, 0.0786, 0.1019, 0.1544])

# 加载预训练的模型权重
model1 = Lstm()
model2 = resnet18()

weight_path1 = './99.1Lstm.pth'
weight_path2 = './resnet18-final.pth'

model1.load_state_dict(torch.load(weight_path1,weights_only=True))
in_channel = model2.fc.in_features
model2.fc = nn.Linear(in_channel, 4)
model2.load_state_dict(torch.load(weight_path2, map_location='cpu',weights_only=True))


def get_marginal_fea(data_list):
    train_img_list = [x[0] for x in data_list]
    train_data_list = [x[1] for x in data_list]
    label = [x[2] for x in data_list]
    num = len(data_list)

    model1.eval()
    with torch.no_grad():
        for idx in range(0, num):
            data = torch.unsqueeze(train_data_list[idx], dim=0)
            output, fea1 = model1(data)
            train_data_list[idx] = torch.squeeze(fea1, dim=0)

    model2.eval()
    with torch.no_grad():
        for idx in range(0, num):
            img = torch.unsqueeze(train_img_list[idx], dim=0)
            output, fea2 = model2(img)
            train_img_list[idx] = torch.squeeze(fea2, dim=0)

    fea_in = []
    for idx in range(0, num):
        fea_in.append(torch.cat((train_img_list[idx], train_data_list[idx])))
    return fea_in, label


class JointLearn(nn.Module):
    def __init__(self, input_dim=640, output_dim=4):
        super(JointLearn, self).__init__()
        self.input = input_dim
        self.output = output_dim
        self.fc1 = nn.Linear(self.input, 160)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(160, self.output)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.drop(x)
        x = self.fc2(x)
        return x


def record_result(pred_list, label_list):
    file_path = 'joint_pred.xlsx'

    mylist = [pred_list, label_list]

    column = ['pred_list', 'label_list']
    test = pd.DataFrame(mylist)
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        test.to_excel(writer, sheet_name='Sheet1', index=False)


class dataset(Dataset):
    def __init__(self, fea, label):
        self.len = len(fea)
        self.fea = fea
        self.label = torch.Tensor(label)

    def __getitem__(self, idx):
        return self.fea[idx], self.label[idx]

    def __len__(self):
        return self.len


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
    all_data_train = PicData(image_train_path, transform=img_transform, data_from=data_train, label_from=label_train)
    all_data_test = PicData(image_test_path, transform=img_transform1, data_from=data_test, label_from=label_test1)

    input_data, label = get_marginal_fea(all_data_train)
    input_test, label_test = get_marginal_fea(all_data_test)
    all_dataset = dataset(input_data, label)
    test_dataset = dataset(input_test, label_test)
    input_train, input_val = torch.utils.data.random_split(all_dataset, [round(0.83*len(all_dataset)),
                                                                         round(0.17*len(all_dataset))])

    train_num = len(input_train)
    test_num = len(input_val)

    batch_size = 128
    train_loader = DataLoader(input_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(input_val, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(test_dataset, 128, shuffle=False)

    net = JointLearn()
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = torch.optim.Adam(net.parameters(), lr)
    epochs = 200
    best_acc = 0.0
    save_path = './{}.pth'.format('JointLearn')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    train_steps = len(train_loader)
    test_steps = len(test_loader)
    trloss = []
    tracc = []
    tloss = []
    tacc = []
    print('start\n')
    # since = time.time()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for epoch in range(epochs):
        # train
        net.train()
        acct = 0.0
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            fea, labels = data

            optimizer.zero_grad()
            fea = torch.Tensor(fea)
            fea = fea.to(device)
            labels = labels.to(device)

            outputs = net(fea)
            predict_x = torch.max(outputs, dim=1)[1]
            acct += torch.eq(predict_x, labels).sum().item()

            loss = loss_function(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_acc = acct / train_num
        tracc.append(train_acc)
        trloss.append(running_loss / train_steps)

        # test
        net.eval()
        acc = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                fea, label_test = test_data

                fea = torch.Tensor(fea)
                fea = fea.to(device)
                label_test = label_test.to(device)
                outputs = net(fea)

                loss = loss_function(outputs, label_test.long())
                test_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, label_test).sum().item()

        scheduler.step()

        test_accurate = acc / test_num
        tloss.append(test_loss / test_steps)
        tacc.append(test_accurate)
        print(
            '[epoch %d] train_ave_loss: %.3f  train_accuracy: %.3f  test_ave_loss: %.3f test_accuracy: %.3f '
            'learning rate: %.8f' %
            (epoch + 1, running_loss / train_steps, train_acc, test_loss / test_steps, test_accurate,
             optimizer.state_dict()['param_groups'][0]['lr']))

        if test_accurate > best_acc:
            best_acc = test_accurate
            torch.save(net.state_dict(), save_path)

    print(best_acc)
    print('Finished Training')
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    #
    # print(start.elapsed_time(end))
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
#     conf_matrix = torch.zeros(4, 4)
    weight_path = './JointLearn.pth'
    net.load_state_dict(torch.load(weight_path))
    pred_list = []
    label_list = []

    net.eval()
    with torch.no_grad():
        for data in val_loader:
            fea, label = data
            label = label.squeeze()
            fea = torch.Tensor(fea)
            fea = fea
            label = label
            out = net(fea)
#           conf_matrix, preds, labels = confusion_matrix(out, label, conf_matrix)
#           pred_list.append(preds.flatten())
#           label_list.append(labels.flatten())
#     conf_matrix = np.array(conf_matrix)
#     df_cm = pd.DataFrame(conf_matrix, index=['Eos', 'Lym', 'Mono', 'Neu'], columns=['Eos', 'Lym', 'Mono', 'Neu'])
#     pp_matrix(df_cm)
#     # record_result(pred_list, label_list)
#     x1 = range(0, epochs)
#     y1 = tracc
#     y2 = trloss
#     y3 = tacc
#     y4 = tloss
#
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(x1, y2, 'ro-', label="train loss")
#     plt.plot(x1, y4, 'bs-', label="test loss")
#     plt.legend()
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#
#     plt.subplot(1, 2, 2)
#     plt.plot(x1, y1, 'ro-', label="train acc")
#     plt.plot(x1, y3, 'bs-', label="test acc")
#     plt.legend()
#     plt.xlabel("epoch")
#     plt.ylabel("acc")
#     plt.legend()
#     plt.show()
#
#
# def confusion_matrix(preds, labels, conf_matrix):
#     preds = torch.max(preds, 1)[1].unsqueeze(dim=-1).numpy()
#     preds = preds.astype(int)
#     labels = labels.unsqueeze(dim=-1).numpy()
#     labels = labels.astype(int)
#     for idx in range(int(len(preds))):
#         conf_matrix[labels[idx, 0], preds[idx, 0]] += 1
#     return conf_matrix, preds, labels


if __name__ == '__main__':
    main()
