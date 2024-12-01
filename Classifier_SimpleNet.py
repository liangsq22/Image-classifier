import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm

# 设置超参数
data_dir = './food_data'  # 数据集路径
batch_size = 64  # 批大小
img_size = 224  # 输入图片大小
num_classes = 10  # 类别数
epochs = 50  # 训练轮数
learning_rate = 0.0001  # 学习率
dropout_rate = 0.3  # Dropout丢弃比率
weight_decay = 1e-4  # L2正则化系数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备

# **1. 数据处理**
# 数据增强和预处理
transform_train = transforms.Compose([
    transforms.Resize((img_size + 98, img_size + 98)), # 缩放图片到比目标裁剪尺寸稍大的大小
    transforms.RandomResizedCrop(img_size),  # 随机裁剪为224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载数据集
dataset = datasets.ImageFolder(root=data_dir)  # 加载图像数据集
class_names = dataset.classes  # 获取类别名称列表
print("Class Names:", class_names)  # 输出类别名称列表

train_size = int(0.8 * len(dataset))  # 划分80%为训练集
val_size = len(dataset) - train_size  # 剩余20%为验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # 随机划分数据集

train_dataset.dataset.transform = transform_train  # 应用训练集的变换
val_dataset.dataset.transform = transform_test  # 应用验证集的变换

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 加载训练数据
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 加载验证数据

# **2. 模型设计：自定义卷积神经网络**
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # 第一层卷积：输入3通道（RGB），输出32通道，卷积核3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 对32通道的特征图进行标准化
        # 第二层卷积：输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 对64通道的特征图进行标准化
        # 第三层卷积：输入64通道，输出128通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # 对128通道的特征图进行标准化
        # 池化层：最大池化，窗口大小2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1：将卷积特征展平后映射到256维
        self.fc1 = nn.Linear(128 * (img_size // 8) * (img_size // 8), 256)
        # 全连接层2：将256维特征映射到分类数
        self.fc2 = nn.Linear(256, num_classes)
        # 激活函数：ReLU（修正线性单元）
        self.relu = nn.ReLU()
        # Dropout层：丢弃比率为30%，用于防止过拟合
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 第一阶段：卷积1 + BatchNorm + ReLU + 池化
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # 第二阶段：卷积2 + BatchNorm + ReLU + 池化
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # 第三阶段：卷积3 + BatchNorm + ReLU + 池化
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # 展平特征：将三维特征图展开为一维向量
        x = x.view(x.size(0), -1)
        # 全连接层1 + ReLU + Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        # 全连接层2（分类层）
        x = self.fc2(x)
        return x

model = CustomCNN(num_classes=num_classes).to(device)  # 模型实例化

# **模型剪枝（Pruning）**
def prune_model(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # 移除剪枝后参数中的掩码

#prune_model(model, amount=0.3)  # 剪枝30%

# **模型量化（Quantization）**
def quantize_model(model):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # 量化配置
    model_fused = torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']])  # 模块融合
    model_prepared = torch.quantization.prepare(model_fused)
    return torch.quantization.convert(model_prepared)

#model = quantize_model(model)

# **3. 损失函数和优化器**
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 使用Adam优化器

# **4. 训练与测试**
def train_and_validate(model, train_loader, val_loader, epochs, criterion, optimizer, device):
    # 存储每轮训练和验证的损失与准确率
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    print("Start training...")  # 开始训练的提示
    for epoch in range(epochs):  # 遍历每一轮训练
        # **训练阶段**
        model.train()  # 将模型设置为训练模式
        train_loss, correct_train = 0, 0  # 初始化训练损失和正确分类计数
        # 使用 tqdm 显示训练进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]", ascii=True)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播，获取模型输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            train_loss += loss.item()  # 累加损失
            _, preds = torch.max(outputs, 1)  # 获取预测类别
            correct_train += (preds == labels).sum().item()  # 累加正确分类的样本数
            train_bar.set_postfix(loss=loss.item())  # 更新进度条显示的损失信息

        # 计算训练集平均损失和准确率
        train_loss /= len(train_loader)
        train_accuracy = correct_train / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

        # **验证阶段**
        model.eval()  # 将模型设置为验证模式
        val_loss, correct_val = 0, 0  # 初始化验证损失和正确分类计数
        all_preds, all_labels = [], []  # 重置每轮的预测值和真实标签
        # 使用 tqdm 显示验证进度条
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]", ascii=True)
        with torch.no_grad():  # 关闭梯度计算，节省内存
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)  # 将数据转移到指定设备
                outputs = model(images)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
                val_loss += loss.item()  # 累加损失
                _, preds = torch.max(outputs, 1)  # 获取预测类别
                correct_val += (preds == labels).sum().item()  # 累加正确分类的样本数
                val_bar.set_postfix(loss=loss.item())  # 更新进度条显示的损失信息
                all_preds.extend(preds.cpu().numpy())  # 保存预测值
                all_labels.extend(labels.cpu().numpy())  # 保存真实标签

        # 计算验证集平均损失和准确率
        val_loss /= len(val_loader)
        val_accuracy = correct_val / len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}: Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # **分类报告和混淆矩阵**
        print("\nClassification Report (Epoch {}):".format(epoch + 1))
        # 生成并打印分类报告
        report = classification_report(all_labels, all_preds, target_names=class_names)
        print(report)

        print("\nConfusion Matrix (Epoch {}):".format(epoch + 1))
        # 设置 Pandas 显示选项，防止矩阵内容被省略
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 500) # 设置显示宽度
        # 生成混淆矩阵并打印
        cm = confusion_matrix(all_labels, all_preds)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        print(cm_df)

        # 记录每轮的损失和准确率
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_accuracy)
        val_acc_list.append(val_accuracy)

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list

# 调用训练与验证函数
train_loss_list, val_loss_list, train_acc_list, val_acc_list = train_and_validate(
    model, train_loader, val_loader, epochs, criterion, optimizer, device
)


# **5. 模型保存**
torch.save(model.state_dict(), "custom_food_classifier.pth")

# **6. 结果可视化**
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_loss_list, label='Train Loss')
plt.plot(range(1, epochs + 1), val_loss_list, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_acc_list, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_acc_list, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.show()

print("Training complete. Model saved as 'custom_food_classifier.pth'.")
