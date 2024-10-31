import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import openpyxl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from feature_lstm_model import Lstm

data_train_path = os.path.abspath('E:/2csj/20230417/Train_main_0811/Result/new_train.mat')
data_test_path = os.path.abspath('E:/2csj/20230417/Train_main_0811/Result/new_test.mat')

# 读取mat文件中的阻抗数据以及标签
data_train_all = scio.loadmat(data_train_path)
data_train = data_train_all['all_cell_train']  # 5578*1 cell
label_train = data_train_all['all_label_train']

data_test_all = scio.loadmat(data_test_path)
data_test = data_test_all['all_cell_test']  # 985*1 cell
label_test1 = data_test_all['all_label_test']

mean = torch.tensor([1.3095e+05, 1.3025e+05, 1.3015e+05, 1.2973e+05, -0.0029, -0.0291, -0.0367, -0.0558])
std = torch.tensor([[6.6195e+03, 6.5976e+03, 6.5766e+03, 6.3026e+03, 0.0076, 0.0779, 0.101, 0.1529]])
#
#
# # 自定义训练集加载函数
# class PicTrainData(Dataset):  # 继承Dataset
#     def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
#         self.root_dir = root_dir  # 文件目录
#         self.transform = transform  # 变换
#         self.data = self.load_img()
#
#     def load_img(self):
#         data_list = []
#         length = len(data_train)
#         for idx in range(length):
#             data = np.array(data_train[idx][0])  # 8*300
#             label = label_train[idx][0]
#
#             x = np.transpose(data)
#             x = torch.Tensor(x)
#             x = x.sub_(mean).div(std)
#             x = torch.Tensor(x)
#             data_list.append((x, label))
#         return data_list
#
#     def __len__(self):  # 返回整个数据集的大小
#         return len(self.data)
#
#     def __getitem__(self, index):  # 根据索引index返回dataset[index]
#         data_info, img_label = self.data[index]  # 根据索引index获取该图片
#         return data_info, img_label  # 返回该样本


class PicData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, data_from, label_from, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.data = self.load_img(data_from, label_from)

    def load_img(self, data_from, label_from):
        data_list = []
        length = len(data_train)
        for idx in range(length):
            data = np.array(data_from[idx][0])  # 8*300
            label = label_from[idx][0]

            x = np.transpose(data)
            x = torch.Tensor(x)
            x = x.sub_(mean).div(std)
            x = torch.Tensor(x)
            data_list.append((x, label))
        return data_list

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_info, data_info, img_label = self.data[index]  # 根据索引index获取该图片
        if self.transform:
            image_info = self.transform(image_info)  # 对样本进行变换
        return image_info, data_info, img_label  # 返回该样本


def main():
    # 图像预处理
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 载入图像及数值数据
    image_train_path = os.path.abspath('E:/2csj/20230417/Train_main_0811/Result/input_train')
    image_test_path = os.path.abspath('E:/2csj/20230417/Train_main_0811/Result/input_test')

    train_dataset = PicData(image_train_path, data_from=data_train, label_from=label_train)
    test_dataset = PicData(image_test_path, data_from=data_test, label_from=label_test1)
    image_train_dataset, image_val_dataset = torch.utils.data.random_split(train_dataset,
                                                                            [round(0.83 * len(train_dataset)),
                                                                             round(0.17 * len(train_dataset))])

    image_train_num = len(image_train_dataset)

    batch_size = 128  # 128
    image_train_loader = DataLoader(image_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # image_test_dataset = PicTestData(image_test_path, transform=image_transform['test'])
    image_val_num = len(image_val_dataset)
    image_test_loader = DataLoader(image_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    image_val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print('dataset completed')
    print("using {} datas for training, {} datas for test.".format(image_train_num, image_val_num))

    net = Lstm()
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    lr = 0.1  # 0.001
    optimizer = optim.Adam(net.parameters(), lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 20*0.5

    epochs = 10
    best_acc = 0.0
    save_path = './{}.pth'.format('Lstm')
    train_steps = len(image_train_loader)
    test_steps = len(image_test_loader)
    trloss = []
    tracc = []
    tloss = []
    tacc = []
    # conf_matrix = torch.zeros(4, 4)
    for epoch in range(epochs):
        # train
        net.train()
        acct = 0.0
        running_loss = 0.0
        for step, data in enumerate(image_train_loader, start=0):
            shu_zhi, labels = data
            optimizer.zero_grad()
            shu_zhi = torch.Tensor(shu_zhi)
            shu_zhi = shu_zhi.to(device)
            labels = labels.to(device)

            outputs, fea = net(shu_zhi)
            predict_x = torch.max(outputs, dim=1)[1]
            acct += torch.eq(predict_x, labels).sum().item()

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_acc = acct / image_train_num
        tracc.append(train_acc)
        trloss.append(running_loss/train_steps)

        # test
        # conf_matrix.zero_()
        net.eval()
        acc = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for step, test_data in enumerate(image_test_loader, start=0):
                shu_zhi_test, label_test = test_data
                # target = label_test.unsqueeze(dim=-1)
                shu_zhi_test = torch.Tensor(shu_zhi_test)
                shu_zhi_test = shu_zhi_test.to(device)
                label_test = label_test.to(device)

                outputs, fea = net(shu_zhi_test)

                loss = loss_function(outputs, label_test)
                test_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                # conf_matrix = confusion_matrix(predict_y, target, conf_matrix)
                acc += torch.eq(predict_y, label_test).sum().item()
        scheduler.step()

        test_accurate = acc / image_val_num
        tloss.append(test_loss/test_steps)
        tacc.append(test_accurate)
        print('[epoch %d] train_ave_loss: %.3f  train_accuracy: %.3f  test_ave_loss: %.3f test_accuracy: %.3f '
              'learning rate: %.8f' %
              (epoch + 1, running_loss / train_steps, train_acc, test_loss/test_steps, test_accurate,
               optimizer.state_dict()['param_groups'][0]['lr']))

        if test_accurate > best_acc:
            best_acc = test_accurate
            torch.save(net.state_dict(), save_path)

    print(best_acc)
    print('Finished Training')
    x1 = range(0, epochs)
    y1 = tracc
    y2 = trloss
    y3 = tacc
    y4 = tloss

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x1, y2, 'ro-', label="train loss")
    plt.plot(x1, y4, 'bs-', label="test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(x1, y1, 'ro-', label="train acc")
    plt.plot(x1, y3, 'bs-', label="test acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
    plt.savefig('./LSTM_result')
    record_result(trloss, tracc, tloss, tacc)

    conf_matrix = torch.zeros(4, 4)

    with torch.no_grad():
        for step, (imgs, targets) in enumerate(image_val_loader):
            targets = targets.squeeze()
            out, fea = net(imgs)
            conf_matrix = confusion_matrix(out, targets, conf_matrix)
    conf_matrix = np.array(conf_matrix)
    print(conf_matrix)
    corrects = conf_matrix.diagonal(offset=0)
    per_kinds = conf_matrix.sum(axis=1)

    labels = ['Eos', 'Lym', 'Mono', 'Neu']
    plt.imshow(conf_matrix, cmap='Blues')

    thresh = conf_matrix.max() / 2
    for x in range(4):
        for y in range(4):
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")
    plt.tight_layout()
    plt.yticks(range(4), labels)
    plt.xticks(range(4), labels, rotation=45)
    plt.show()


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.max(preds, 1)[1].unsqueeze(dim=-1).numpy()
    labels = labels.unsqueeze(dim=-1).numpy()
    for idx in range(len(preds)):
        conf_matrix[preds[idx, 0], labels[idx, 0]] += 1
    return conf_matrix


def record_result(trloss, tracc, tloss, tacc):
    file_path = 'result'
    batch = '128'
    drop = '0.1'
    l2 = '1e-6'
    initial_lr = '0.01'
    schedular = '30_0.5'
    path = drop + '#' + l2 + '#' + batch + '#' + initial_lr + '#' + schedular + file_path + '.xlsx'

    mylist = [trloss, tracc, tloss, tacc]

    column = ['train_loss', 'train_acc', 'test_loss', 'test_acc']
    test = pd.DataFrame(mylist)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        test.to_excel(writer, sheet_name='Sheet1', index=False)


if __name__ == '__main__':
    main()
