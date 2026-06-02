import copy
import time
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
from model import VGG16
import pandas as pd
import matplotlib.pyplot as plt

# 1. 获取 FashionMNIST 数据，并划分为训练集和验证集
# FashionMNIST 的 train=True 一共有 60000 张训练图片
# 这里按 8:2 划分：约 48000 张用于训练，约 12000 张用于验证
def train_val_data_process():
    # 下载/读取 FashionMNIST 训练集
    train_data = FashionMNIST(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor()
        ]),
        download=True
    )

    # 把 60000 个样本随机划分成训练集和验证集
    # train_data 约有 48000 个样本，val_data 约有 12000 个样本
    train_data, val_data = Data.random_split(
        train_data,
        [round(0.8 * len(train_data)), round(0.2 * len(train_data))]
    )

    # 训练集 DataLoader
    # batch_size=? 表示每次从训练集中取 ? 张图片组成一个 batch
    # shuffle=True 表示每个 Epoch 开始前打乱训练样本顺序，有利于训练
    train_dataloader = Data.DataLoader(
        dataset=train_data,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    # 验证集 DataLoader
    # batch_size=? 表示每次从验证集中取 ? 张图片组成一个 batch
    # 验证阶段不更新参数，所以 shuffle=False 更常见，方便结果稳定复现
    val_dataloader = Data.DataLoader(
        dataset=val_data,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    return train_dataloader, val_dataloader


# 2. 模型训练和验证过程
# num_epochs 表示整个训练集会被模型完整学习多少轮
# 例如 num_epochs=10，表示 48000 张训练样本会被重复训练 10 轮
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 如果电脑支持 CUDA，就把模型和数据放到 GPU；否则放到 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Adam 优化器负责根据反向传播得到的梯度更新模型参数
    # lr=0.001 是学习率，控制每次参数更新的步长
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 交叉熵损失函数用于多分类任务
    # 对于一个 batch，它会计算这个 batch 中所有样本的平均分类损失
    criterion = nn.CrossEntropyLoss()

    # 把模型参数移动到 GPU 或 CPU
    model = model.to(device)

    # 保存当前模型参数
    # 后面如果某个 Epoch 的验证集准确率最高，就更新这份参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # best_acc 用来记录所有 Epoch 中验证集准确率最高的那一次
    best_acc = 0.0

    # 每个元素保存一个 Epoch 的训练集平均 loss
    train_loss_all = []

    # 每个元素保存一个 Epoch 的验证集平均 loss
    val_loss_all = []

    # 每个元素保存一个 Epoch 的训练集准确率
    train_acc_all = []

    # 每个元素保存一个 Epoch 的验证集准确率
    val_acc_all = []

    # 记录训练开始时间，用来计算总耗时
    since = time.time()

    # epoch 表示当前是第几轮训练
    # 一个 Epoch = 模型完整看完一遍训练集中的所有样本
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # train_loss 累加当前 Epoch 中所有训练样本的 loss 总和
        # 注意：不是一个 batch 的 loss，而是整个 Epoch 的累计 loss
        train_loss = 0.0

        # train_corrects 累加当前 Epoch 中预测正确的训练样本数量
        train_corrects = 0

        # val_loss 累加当前 Epoch 中所有验证样本的 loss 总和
        val_loss = 0.0

        # val_corrects 累加当前 Epoch 中预测正确的验证样本数量
        val_corrects = 0

        # train_num 累加当前 Epoch 已经处理过的训练样本总数
        train_num = 0

        # val_num 累加当前 Epoch 已经处理过的验证样本总数
        val_num = 0

        # =========================
        # 2.1 训练阶段
        # =========================

        # 设置模型为训练模式
        # 如果模型中有 Dropout、BatchNorm，它们会按训练方式工作
        model.train()

        # train_dataloader 每次返回一个 batch
        # b_x 是一个 batch 的图片，形状通常是 [batch_size, 1, 224, 224]
        # b_y 是这个 batch 对应的真实标签，形状通常是 [batch_size]
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 当前 batch 的实际样本数
            batch_size = b_x.size(0)

            # 把当前 batch 的图片移动到 GPU 或 CPU
            b_x = b_x.to(device)

            # 把当前 batch 的标签移动到 GPU 或 CPU
            b_y = b_y.to(device)

            # 前向传播
            # 输入一个 batch 的图片，输出这个 batch 中每张图片属于各类别的得分
            # output 形状通常是 [batch_size, 10]
            output = model(b_x)

            # 对每个样本，取 10 个类别得分中最大的那个类别作为预测类别
            # pre_lab 形状是 [batch_size]
            pre_lab = torch.argmax(output, dim=1)

            # 计算当前 batch 的平均 loss
            # criterion 默认返回这个 batch 内所有样本 loss 的平均值
            loss = criterion(output, b_y)

            # 清空上一轮 batch 残留的梯度
            # PyTorch 默认会累加梯度，所以每个 batch 反向传播前都要清零
            optimizer.zero_grad()

            # 反向传播
            # 根据当前 batch 的 loss 计算每个参数的梯度
            loss.backward()

            # 参数更新
            # 优化器根据梯度调整模型参数，这是训练阶段真正“学习”的一步
            optimizer.step()

            # 把当前 batch 的平均 loss 还原成当前 batch 的 loss 总和，再累加到当前 Epoch
            # loss.item() 是 batch 平均 loss
            # batch_size 是当前 batch 样本数128
            train_loss += loss.item() * batch_size

            # 累加当前 batch 中预测正确的样本数量
            train_corrects += torch.sum(pre_lab == b_y).item()

            # 累加当前 Epoch 已经训练过的样本数量
            train_num += batch_size

        # =========================
        # 2.2 验证阶段
        # =========================

        # 设置模型为验证/评估模式
        # 如果模型中有 Dropout、BatchNorm，它们会按验证方式工作
        model.eval()

        # 验证阶段不需要计算梯度
        # 这样可以节省显存和计算时间，也避免误更新参数
        with torch.no_grad():
            # val_dataloader 每次返回一个验证 batch
            # 验证集只用于评估模型效果，不参与参数更新
            for step, (b_x, b_y) in enumerate(val_dataloader):
                # 当前验证 batch 的实际样本数
                batch_size = b_x.size(0)

                # 把当前 batch 的验证图片移动到 GPU 或 CPU
                b_x = b_x.to(device)

                # 把当前 batch 的验证标签移动到 GPU 或 CPU
                b_y = b_y.to(device)

                # 前向传播
                # 验证阶段也要得到预测结果，但不会反向传播
                output = model(b_x)

                # 对每个验证样本，取分数最大的类别作为预测类别
                pre_lab = torch.argmax(output, dim=1)

                # 计算当前验证 batch 的平均 loss
                loss = criterion(output, b_y)

                # 把当前验证 batch 的平均 loss 转换成 loss 总和，并累加到当前 Epoch
                val_loss += loss.item() * batch_size

                # 累加当前验证 batch 中预测正确的样本数量
                val_corrects += torch.sum(pre_lab == b_y).item()

                # 累加当前 Epoch 已经验证过的样本数量
                val_num += batch_size

        # =========================
        # 2.3 统计当前 Epoch 的结果
        # =========================

        # 当前 Epoch 的训练集平均 loss
        # 等于这个 Epoch 所有训练样本的 loss 总和 / 训练样本总数
        epoch_train_loss = train_loss / train_num

        # 当前 Epoch 的训练集准确率
        # 等于预测正确的训练样本数 / 训练样本总数
        epoch_train_acc = train_corrects / train_num

        # 当前 Epoch 的验证集平均 loss
        # 等于这个 Epoch 所有验证样本的 loss 总和 / 验证样本总数
        epoch_val_loss = val_loss / val_num

        # 当前 Epoch 的验证集准确率
        # 等于预测正确的验证样本数 / 验证样本总数
        epoch_val_acc = val_corrects / val_num

        # 每个 Epoch 只保存一次训练 loss
        train_loss_all.append(epoch_train_loss)

        # 每个 Epoch 只保存一次训练准确率
        train_acc_all.append(epoch_train_acc)

        # 每个 Epoch 只保存一次验证 loss
        val_loss_all.append(epoch_val_loss)

        # 每个 Epoch 只保存一次验证准确率
        val_acc_all.append(epoch_val_acc)

        print("{} train loss: {:.4f} train acc: {:.4f}".format(epoch + 1, epoch_train_loss, epoch_train_acc))
        print("{} val loss: {:.4f} val acc: {:.4f}".format(epoch + 1, epoch_val_loss, epoch_val_acc))

        # 如果当前 Epoch 的验证准确率超过历史最好结果，就保存当前模型参数
        # 验证集准确率更能反映模型对未见样本的泛化能力
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # 计算所有 Epoch 的总训练时间
    time_use = time.time() - since
    print("训练和验证耗费的时间: {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    # 保存最高准确率下的模型参数到一个文件里
    torch.save(best_model_wts,'E:/PythonProject/VGG16/best_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all, })

    return train_process

def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载模型
    model = VGG16()
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练,把结果存到train_process
    train_process = train_model_process(model, train_data, val_data, num_epochs=20)
    # 把train_process内容可视化
    matplot_acc_loss(train_process)