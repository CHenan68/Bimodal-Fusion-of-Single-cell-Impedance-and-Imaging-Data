import os
import numpy as np
import scipy.io as scio
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
# from pretty_confusion_matrix import pp_matrix_from_data
import matplotlib.pyplot as plt
from lstm_model import Lstm
from resnet18_model import resnet18
from torch.utils.data import Dataset
from tSNE import plot_tsne
import time


data_train_path = os.path.abspath('D:/毕设工作/new_train.mat')
data_test_path = os.path.abspath('D:/毕设工作/new_test.mat')

# 读取mat文件中的阻抗数据以及标签
data_train_all = scio.loadmat(data_train_path)
data_train = data_train_all['all_cell_train']  # 5578*1 cell
label_train = data_train_all['all_label_train']

data_test_all = scio.loadmat(data_test_path)
data_test = data_test_all['all_cell_test']  # 5578*1 cell
label_test = data_test_all['all_label_test']

# 加载预训练的模型权重
model1 = Lstm()
model2 = resnet18()

weight_path1 = './99.1Lstm.pth'
weight_path2 = './resnet18-final.pth'

mean = torch.tensor([1.3106e+05, 1.3036e+05, 1.3027e+05, 1.2984e+05, -0.0029, -0.0292, -0.0369, -0.0561])
std = torch.tensor([6.6278e+03, 6.604e+03, 6.5823e+03, 6.3055e+03, 0.0075, 0.0786, 0.1019, 0.1544])

model1.load_state_dict(torch.load(weight_path1))
in_channel = model2.fc.in_features
model2.fc = nn.Linear(in_channel, 4)
model2.load_state_dict(torch.load(weight_path2, map_location='cpu',weights_only=True))


def main():
    img_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.037, 0.037, 0.037], [0.0105, 0.0105, 0.0105])
                                        ])
    img_transform1 = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.037, 0.037, 0.037], [0.0105, 0.0105, 0.0105])
                                        ])

    image_train_path = os.path.abspath('D:/毕设工作/input_train')  # E:/2csj/20230417/Train_main_0811/Result/input_train
    image_test_path = os.path.abspath('D:/毕设工作/input_test')
    # 加载数据集
    # all_data = PicData(image_train_path, transform=img_transform, data_from=data_train, label_from=label_train)
    all_dest = PicData(image_test_path, transform=img_transform1, data_from=data_test, label_from=label_test)
    # all_data_train, all_data_val = torch.utils.data.random_split(all_data, [round(0.83 * len(all_data)),
    #                                                        round(0.17 * len(all_data))])
    #  num_train = len(all_data_train)
    # num_val = len(all_data_val)
    num_test = len(all_dest)


    # out_train, pred_train, true_train = dec_ave_fusion(all_data_train)
    # out_test, pred_test, true_test = dec_ave_fusion(all_data_val)
    out_val, pred_val, true_val = dec_ave_fusion(all_dest)
    # fea = torch.tensor(out_test)
    #  plot_tsne(fea, true_test)

    # accuracy_train = torch.eq(torch.Tensor(pred_train), torch.Tensor(true_train)).sum().item() / num_train
    # accuracy_test = torch.eq(torch.Tensor(pred_test), torch.Tensor(true_test)).sum().item() / num_val
    accuracy_val = torch.eq(torch.Tensor(pred_val), torch.Tensor(true_val)).sum().item() / num_test

    # print('after fusion:\n')
    # print('train:', accuracy_train)
    # print('val', accuracy_test)
    # print('test', accuracy_val)

    y_test = np.array(true_val)
    y_test_pred = np.array(pred_val)
    # record_result(y_test_pred, y_test)
    # pp_matrix_from_data(y_test, y_test_pred, columns=['Eos', 'Lym', 'Mono', 'Neu'])


def dec_ave_fusion(data):
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


    pred = []
    pred_list = []
    for idx in range(0, num):
        dec_fusion = 0.5*data_list[idx]+img_list[idx]*0.5
        pred_list.append(np.array(dec_fusion))
        pred_y = torch.max(dec_fusion, dim=0)[1]
        pred.append(pred_y)
    return pred_list, pred, label


def record_result(pred_list, label_list):
    file_path = 'ave_pred.xlsx'
    pred_list = pred_list.ravel().tolist()
    label_list = label_list.ravel().tolist()

    mylist = [pred_list, label_list]

    column = ['pred_list', 'label_list']
    test = pd.DataFrame(mylist)
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        test.to_excel(writer, sheet_name='Sheet1', index=False)


class PicData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform, data_from, label_from):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.data = self.load_img(data_from, label_from)

    def load_img(self, data_from, label_from):
        path_list = list(filter(os.path.isdir,
                                map(lambda filename: os.path.join(self.root_dir, filename),
                                    os.listdir(self.root_dir))
                                ))
        data_list = []
        idx = 0
        idy = 0
        for path in path_list:
            imgs = os.listdir(path)  # 目录里的所有文件
            for im in imgs:
                img_path = os.path.join(path, im)  # 获取索引为index的图片的路径名
                img = Image.open(img_path)  # 读取该图片

                data = np.array(data_from[idx][0])  # 8*300
                label = label_from[idx][0]

                x = np.transpose(data)
                x = torch.Tensor(x)
                x = x.sub_(mean).div(std)
                x = torch.Tensor(x)

                data_list.append((img, x, idy))
                idx += 1
            idy += 1
        return data_list

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_info, data_info, img_label = self.data[index]  # 根据索引index获取该图片
        if self.transform:
            image_info = self.transform(image_info)  # 对样本进行变换
        return image_info, data_info, img_label  # 返回该样本


if __name__ == '__main__':
    main()

