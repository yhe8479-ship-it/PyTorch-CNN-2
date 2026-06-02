import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import GoogLeNet,Inception

# 1. 获取 FashionMNIST 测试集
# 对照 train 代码：
# train 代码里 train=True，表示读取 60000 张训练图片
# test 代码里 train=False，表示读取 10000 张测试图片
def test_data_process():
    # 下载/读取 FashionMNIST 测试集
    # 测试集不会参与训练，只用来评估模型最终效果
    test_data = FashionMNIST(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor()
        ]),
        download=True
    )

    # 测试集 DataLoader
    # 对照 train 代码：
    # train_dataloader 是训练 batch，用于更新模型参数
    # test_dataloader 是测试 batch，只用于计算准确率，不更新参数
    test_dataloader = Data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    return test_dataloader

# 2. 模型测试过程
# 对照 train_model_process：
# 训练阶段：前向传播 + 计算 loss + 反向传播 + 更新参数
# 测试阶段：只做前向传播 + 统计预测正确的样本数
def test_model_process(model, test_dataloader):
    # 如果电脑支持 CUDA，就用 GPU；否则用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 把模型移动到 GPU 或 CPU
    model = model.to(device)

    # 设置模型为评估模式
    model.eval()

    # test_corrects 用来累计测试集中预测正确的样本数量
    test_corrects = 0

    # test_num 用来累计已经测试过的样本数量
    test_num = 0

    # 测试阶段不需要计算梯度
    # 因为测试集只评估模型效果，不更新模型参数
    with torch.no_grad():
        # test_dataloader 每次返回一个测试 batch
        for test_data_x, test_data_y in test_dataloader:

            batch_size = test_data_x.size(0)

            # 把当前 batch 的图片移动到 GPU 或 CPU
            test_data_x = test_data_x.to(device)

            # 把当前 batch 的标签移动到 GPU 或 CPU
            test_data_y = test_data_y.to(device)

            # 前向传播
            # 输入一个 batch 的测试图片
            # 输出每张图片属于 10 个类别的得分
            # output 形状通常是 [batch_size, 10]
            output = model(test_data_x)

            # 对每个样本，取 10 个类别得分中最大的类别作为预测类别
            # pre_lab 形状是 [batch_size]
            pre_lab = torch.argmax(output, dim=1)

            # 累加当前 batch 中预测正确的样本数量
            test_corrects += torch.sum(pre_lab == test_data_y).item()

            # 累加当前已经测试过的样本数量
            test_num += batch_size

    # 测试集最终准确率
    # 等于预测正确的测试样本数 / 测试样本总数
    test_acc = test_corrects / test_num

    print("测试样本总数:", test_num)
    print("预测正确样本数:", test_corrects)
    print("测试集准确率为: {:.4f}".format(test_acc))

if __name__=="__main__":
    # 加载模型
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load('E:/PythonProject/GoogLeNet/best_model.pth'))
    # 加载数据
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    # 1.直接得到准确率结果
    test_model_process(model,test_dataloader)

    # 2.显示具体预测+真实
    # device = "cuda" if torch.cuda.is_available() else 'cpu'
    # model = model.to(device)
    #
    # classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    # with torch.no_grad():
    #     for b_x,b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #
    #         model.eval()
    #         output = model(b_x)
    #         pre_lab = torch.argmax(output,dim=1)
    #         # pre_lab和b_y原本都是tensor类型，通过item()转化成类别编号,可以作为下标，把编号转换成名称
    #         result = pre_lab.item()
    #         label = b_y.item()
    #
    #         print("预测值: ",classes[result],"------","真实值: ",classes[label])
