import time

import numpy as np
import torch
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
from torchvision import transforms, datasets
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data_level_model1 import resnet18
# from pretty_confusion_matrix import pp_matrix


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载图像数据路径
    cell_train_path = os.path.abspath('D:/毕设工作/input_train')
    amp_train_path = os.path.abspath('D:/毕设工作/imp_amp_train')
    pha_train_path = os.path.abspath('D:/毕设工作/imp_pha_train')

    # 将图像转为单通道灰度图，并于标签一起加载到一个链表中
    cell_train = load_img(cell_train_path)
    amp_train = load_img(amp_train_path)
    pha_train = load_img(pha_train_path)
    cell_test_path = os.path.abspath('D:/毕设工作/input_test')
    amp_test_path = os.path.abspath('D:/毕设工作/imp_amp_test')
    pha_test_path = os.path.abspath('D:/毕设工作/imp_pha_test')

    # 将图像转为单通道灰度图，并于标签一起加载到一个链表中
    cell_test = load_img(cell_test_path)
    amp_test = load_img(amp_test_path)
    pha_test = load_img(pha_test_path)

    # 将上一步得到的三个链表中的图像部分分别作为R、G、B通道合并为一个三通道图像，并以链表形式输出
    input_train, label_train = cat_pic(cell_train, amp_train, pha_train)
    input_test, label_test = cat_pic(cell_test, amp_test, pha_test)
    # 计算图像数据集各通道的均值、方差，为标准化做准备
    # mean, std = cal_mean_std(input_test)

    cat_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        # transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.0364, 0.3207, 0.3205], [0.0104, 0.1387, 0.1385])
                                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

    # 制作dataset，并划分训练集、测试集
    train_dataset = my_dataset(input_train, label_train, transform=cat_transform)
    test_dataset = my_dataset(input_test, label_test, transform=cat_transform)
    train_input, val_input = torch.utils.data.random_split(train_dataset, [round(0.83 * len(train_dataset)),
                                                           round(0.17 * len(train_dataset))])

    train_num = len(train_input)
    test_num = len(val_input)

    batch_size = 128

    train_loader = DataLoader(train_input, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_input, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(test_dataset, 128, shuffle=False)
    print("Dataset completed!")

    # 加载网络结构与训练超参数
    net = resnet18()
    model_weight_path = "./resnet18-pre.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, weights_only=True))

    # 迁移学习，更改fc层
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 4)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    lr = 0.0001
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epochs = 20
    best_acc = 0.0
    save_path = './{}.pth'.format('DirectCnn_small1030')
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
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            predict_x = torch.max(outputs, dim=1)[1]
            acct += torch.eq(predict_x, labels).sum().item()

            loss = loss_function(outputs, labels)
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
                image_test, label_test = test_data
                image_test = image_test.to(device)
                label_test = label_test.to(device)

                outputs = net(image_test)
                loss = loss_function(outputs, label_test)
                test_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, label_test).sum().item()
        # scheduler.step()

        test_accurate = acc / test_num
        tloss.append(test_loss / test_steps)
        tacc.append(test_accurate)
        print(
            '[epoch %d] train_ave_loss: %.3f  train_accuracy: %.3f  test_ave_loss: %.3f test_accuracy: %.3f learning '
            'rate: %.8f' %
            (epoch + 1, running_loss / train_steps, train_acc, test_loss / test_steps, test_accurate,
             optimizer.state_dict()['param_groups'][0]['lr']))

        if test_accurate > best_acc:
            best_acc = test_accurate
            torch.save(net.state_dict(), save_path)

    print("Best Accuracy:", best_acc)
    print('Finished Training')

    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(start.elapsed_time(end))

    # conf_matrix = torch.zeros(4, 4)
    # weight = './DirectCnn.pth'
    # net.load_state_dict(torch.load(weight))
    # since = time.time()
    # with torch.no_grad():
    #     for data in val_loader:
    #         fea, label = data
    #         label = label.squeeze()
    #         fea = torch.Tensor(fea)
    #         out = net(fea)
    # time_elapsed = time.time() - since
    # print(time_elapsed)
    #         conf_matrix = confusion_matrix(out, label, conf_matrix)
    # conf_matrix = np.array(conf_matrix)
    # df_cm = pd.DataFrame(conf_matrix, index=['Eos', 'Lym', 'Mono', 'Neu'], columns=['Eos', 'Lym', 'Mono', 'Neu'])
    # pp_matrix(df_cm)
    # plot_matrix(conf_matrix, [0, 1, 2, 3], title='confusion_matrix_svc',
    #             axis_labels=['Eos', 'Lym', 'Mono', 'Neu'])
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

    # record_result(trloss, tracc, tloss, tacc)


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.max(preds, 1)[1].unsqueeze(dim=-1).numpy()
    preds = preds.astype(int)
    labels = labels.unsqueeze(dim=-1).numpy()
    labels = labels.astype(int)
    for idx in range(int(len(preds))):
        conf_matrix[labels[idx, 0], preds[idx, 0]] += 1
    return conf_matrix


def record_result(trloss, tracc, tloss, tacc):
    file_path = 'Direct_result1'
    l2 = '0'
    batch = '128'
    initial_lr = '0.0001'
    schedular = '0'
    path = l2 + '#' + batch + '#' + initial_lr + '#' + schedular + file_path + '.xlsx'

    mylist = [trloss, tracc, tloss, tacc]

    column = ['train_loss', 'train_acc', 'test_loss', 'test_acc']
    test = pd.DataFrame(mylist)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        test.to_excel(writer, sheet_name='Sheet1', index=False)


# 单种图像加载函数
def load_img(root_dir):
    path_list = list(filter(os.path.isdir,
                            map(lambda filename: os.path.join(root_dir, filename),
                                os.listdir(root_dir))
                            ))
    img_list = []
    idx = 0
    label = 0
    for path in path_list:
        imgs = os.listdir(path)  # 目录里的所有文件
        for im in imgs:
            img_path = os.path.join(path, im)  # 获取索引为index的图片的路径名
            img = Image.open(img_path)  # 读取该图片
            if img.mode != 'L':
                img = img.convert('L')
            # img_arr = np.array(img, dtype=np.float32) / 255
            img = img.resize((224, 224))
            # img = np.array(img, dtype=np.float32)
            img_list.append((img, label))
            idx += 1
        label += 1
    return img_list


# 三种图像拼接函数
def cat_pic(cell, amp, pha):
    num = len(cell)
    cat = []
    cell_list = [x[0] for x in cell]
    amp_list = [x[0] for x in amp]
    pha_list = [x[0] for x in pha]
    label = [x[1] for x in cell]

    for idx in range(0, num):
        rgb = np.zeros((224, 224, 3))
        r = np.array(cell_list[idx])
        g = np.array(amp_list[idx])
        b = np.array(pha_list[idx])
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        cat.append(Image.fromarray(np.uint8(rgb)))

    return cat, label


# 将链表转为dataset类
class my_dataset(Dataset):
    def __init__(self, cat, label, transform):
        self.len = len(cat)
        self.cat = cat
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        cat1 = self.cat[idx]
        cat1 = self.transform(cat1)
        return cat1, self.label[idx]

    def __len__(self):
        return self.len


# 计算图像各通道的均值、标准差
def cal_mean_std(datalist):
    total_pixels = 0
    sum_normalized_pixel_values = np.zeros(3)
    num = len(datalist)
    for idx in range(num):
        image = datalist[idx]
        image_array = np.array(image)
        normalized_image_array = image_array / 255.0

        total_pixels += normalized_image_array.size
        sum_normalized_pixel_values += np.sum(normalized_image_array, axis=(0, 1))

    mean = sum_normalized_pixel_values / total_pixels
    mean_out = torch.Tensor(mean)

    sum_squared_diff = np.zeros(3)
    for idx in range(num):
        image = datalist[idx]
        image_array = np.array(image)
        normalized_image_array = image_array / 255.0
        try:
            diff = (normalized_image_array - mean) ** 2
            sum_squared_diff += np.sum(diff, axis=(0, 1))
        except:
            print(f"捕获到自定义异常")

    variance = sum_squared_diff / total_pixels
    std = torch.Tensor(variance)

    print("Mean:", mean_out)
    print("Variance:", std)

    return mean_out, std


if __name__ == '__main__':
    main()





