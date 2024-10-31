import os
import json
import sys
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from draw_cam import draw_CAM
from feature_resnet18_model import resnet18
from cal_cam import cal_cam

def main():
    # print(torch.__version__)
    # 数据集加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    img_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.037, 0.037, 0.037], [0.0105, 0.0105, 0.0105])
                                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        "test": transforms.Compose([transforms.Resize((224, 224)),
                                    # transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.03688, 0.03688, 0.03688], [0.01035, 0.1035, 0.01035])
                                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])}
    # 数据集加载
    im_root = os.path.abspath(os.path.join(os.getcwd(), "../."))
    train_path = os.path.join(im_root, "cell_im_train")
    # test_path = os.path.join(im_root, "data", "test")
    # 获取所有用于训练的样本
    train_data = datasets.ImageFolder(root=train_path, transform=img_transform["train"])
    # 9：1划分训练集和测试集
    image_train_dataset, image_test_dataset = torch.utils.data.random_split(train_data, [round(0.8*len(train_data)),
                                                                                         round(0.2*len(train_data))])
    image_train_num = len(image_train_dataset)

    batch_size = 64

    # 标签json文件生成
    cell_list = train_data.class_to_idx
    cla_dict = dict((val, key) for key, val in cell_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=3)
    with open('cell_im_classes.json', 'w') as json_file:
        json_file.write(json_str)

    train_loader = torch.utils.data.DataLoader(image_train_dataset, batch_size=batch_size, shuffle=True)

    # image_test_dataset = datasets.ImageFolder(root=test_path, transform=img_transform["test"])
    test_loader = torch.utils.data.DataLoader(image_test_dataset, batch_size=batch_size, shuffle=False)
    image_test_num = len(image_test_dataset)

    print('dataset completed')
    print("using {} images for training, {} images for validation.".format(image_train_num,
                                                                           image_test_num))

    # 训练参数设置
    net = resnet18()
    model_weight_path = "./resnet18-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, weights_only=True))
    # for param in net.parameters():
    #     param.requires_grad = False

    # 迁移学习，更改fc层
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 4)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    lr = 0.01
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)  # factor=0.6
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epochs = 50
    best_acc = 0.0
    save_path = './{}.pth'.format('resnet1030')
    train_steps = len(train_loader)
    test_steps = len(test_loader)
    # 用于记录训练中loss和acc以绘图
    trloss = []
    tracc = []
    tloss = []
    tacc = []
    # 训练
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    # Waits for everything to finish running

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
            outputs, fea = net(images)
            predict_x = torch.max(outputs, dim=1)[1]
            acct += torch.eq(predict_x, labels).sum().item()

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_acc = acct / image_train_num
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
                outputs, fea = net(image_test)
                loss = loss_function(outputs, label_test)
                test_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, label_test).sum().item()
        # scheduler.step(test_loss / test_steps)

        test_accurate = acc / image_test_num
        tloss.append(test_loss / test_steps)
        tacc.append(test_accurate)
        print(
            '[epoch %d] train_ave_loss: %.3f  train_accuracy: %.3f  test_ave_loss: %.3f test_accuracy: '
            '%.3f learning rate: %.8f' %
            (epoch + 1, running_loss / train_steps, train_acc, test_loss / test_steps, test_accurate,
             optimizer.state_dict()['param_groups'][0]['lr']))

        scheduler.step()

        if test_accurate > best_acc:
            best_acc = test_accurate
            torch.save(net.state_dict(), save_path)

    print(best_acc)
    print('Finished Training')
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
    # impath = os.path.abspath('./0002.jpg')
    # savepath = os.path.abspath('D:/multimodal_fusion')
    # net.to("cpu")
    # cam = cal_cam()

    # draw_CAM(net, impath,savepath,img_transform["train"],True)
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


if __name__ == '__main__':
    main()
