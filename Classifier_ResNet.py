import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# 输出 utf8 格式
sys.stdout.reconfigure(encoding='utf-8')

# 设置预训练模型的缓存路径
os.environ['TORCH_HOME'] = 'D:/torch_cache' 

# 设置超参数
data_dir = './food_data'  # 数据集路径
batch_size = 64 # 批大小
img_size = 224 # 裁剪图像大小，224公认为较好的尺寸用于训练
num_classes = 10 # 分类数
epochs = 20 # 训练轮数
learning_rate = 0.0001 # 学习率
dropout_rate = 0.5  # Dropout层丢弃率
weight_decay = 1e-3  # L2 正则化率
scheduler_patience = 3  # 学习率调度器的等待轮数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **1. 数据处理**
# 数据增强和预处理
transform_train = transforms.Compose([
    transforms.Resize((img_size + 98, img_size + 98)), # 缩放图片到比目标裁剪尺寸稍大的大小
    transforms.RandomResizedCrop(img_size), # 随机裁剪
    transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 随机颜色变换
    transforms.ToTensor(), # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
])

transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)), # 调整大小
    transforms.ToTensor(), # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
])

# 加载数据集
dataset = datasets.ImageFolder(root=data_dir)
class_names = dataset.classes  # 获取类别名称列表
print("Class Names:", class_names)  # 输出类别名称列表

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = transform_train
val_dataset.dataset.transform = transform_test

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# **2-1. 模型设计：参考ResNet18进行模仿和简化**
class SimplifiedResNet(nn.Module):
    def __init__(self, num_classes):
        super(SimplifiedResNet, self).__init__()
        # 第一层：初始卷积层和池化层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 输入：3通道 (RGB)，输出：64通道
            nn.BatchNorm2d(64),  # Batch Normalization标准化，提升训练速度和稳定性
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 降低特征图大小
        )
        # 残差模块1：输入64通道，输出64通道
        self.layer1 = self._make_layer(64, 64, 2)
        # 残差模块2：输入64通道，输出128通道
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        # 残差模块3：输入128通道，输出256通道
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        # 残差模块4：输入256通道，输出512通道
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        # 全局平均池化层和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 特征图缩小为1x1大小
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout层
        self.fc = nn.Linear(512, num_classes)  # 分类层，输入512维，输出num_classes

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # 第一个残差块：调整通道数和分辨率
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # 剩余的残差块
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 初始卷积
        x = self.layer1(x)  # 第一阶段
        x = self.layer2(x)  # 第二阶段
        x = self.layer3(x)  # 第三阶段
        x = self.layer4(x)  # 第四阶段
        x = self.avgpool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展平成向量
        x = self.dropout(x)  # Dropout层
        x = self.fc(x)  # 分类
        return x

# 残差块的定义
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 主路径：两个3x3卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 残差连接路径：调整输入维度匹配输出（当stride或通道数不同）
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x  # 保存输入，用于残差连接
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 如果需要调整残差连接的维度
        if self.downsample is not None:
            identity = self.downsample(x)
        # 残差连接
        out += identity
        out = self.relu(out)
        return out


# **2-2. 模型设计：基于预训练的ResNet18**
class FoodClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FoodClassifier, self).__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 加载预训练模型
        # 冻结并解冻特定层
        for param in self.base_model.parameters():
            param.requires_grad = False
        # 解冻特定的层（ layer4 和 fc 层）
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True
        # 替换分类头
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Dropout层
            nn.Linear(self.base_model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# model = SimplifiedResNet(num_classes=num_classes).to(device)
model = FoodClassifier(num_classes=num_classes).to(device) # 模型实例化

# **模型剪枝（Pruning）**
def prune_model(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # 移除剪枝后参数中的掩码

#prune_model(model, amount=0.5)  # 剪枝50%

# **模型量化（Quantization）**
def quantize_model(model):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # 量化配置
    model_fused = torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']])  # 模块融合
    model_prepared = torch.quantization.prepare(model_fused)
    return torch.quantization.convert(model_prepared)

#model = quantize_model(model)

# **3. 损失函数和优化器**
criterion = nn.CrossEntropyLoss() # 使用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # 使用Adam优化器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=scheduler_patience) # 学习率调度器

# **4. 训练与测试**
def train_and_validate(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler, device):
    # 用于存储训练和验证集的损失值和准确率
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    print("Start training...")
    for epoch in range(epochs):
        # **训练阶段**
        model.train()  # 设置模型为训练模式
        train_loss, correct_train = 0, 0  # 初始化训练损失和正确分类计数

        # tqdm 进度条用于实时显示训练进度
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]", ascii=True)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)  # 将数据加载到设备（GPU/CPU）
            optimizer.zero_grad()  # 清除上一轮的梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            train_loss += loss.item()  # 累加损失
            _, preds = torch.max(outputs, 1)  # 获取预测值
            correct_train += (preds == labels.data).sum().item()  # 累加正确分类的样本数

            # 更新进度条上的当前损失值
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 计算本轮训练的平均损失和准确率
        train_loss /= len(train_loader)
        train_accuracy = correct_train / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

        # **验证阶段**
        model.eval()  # 设置模型为验证模式
        val_loss, correct_val = 0, 0  # 初始化验证损失和正确分类计数
        all_preds, all_labels = [], []  # 重置预测值和真实标签

        # tqdm 进度条用于实时显示验证进度
        test_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]", ascii=True)
        with torch.no_grad():  # 禁用梯度计算，节省内存
            for images, labels in test_bar:
                images, labels = images.to(device), labels.to(device)  # 将数据加载到设备
                outputs = model(images)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
                val_loss += loss.item()  # 累加损失
                _, preds = torch.max(outputs, 1)  # 获取预测值
                correct_val += (preds == labels.data).sum().item()  # 累加正确分类的样本数
                test_bar.set_postfix(loss=f"{loss.item():.4f}")  # 更新进度条上的当前损失值

                # 保存当前 batch 的预测值和真实标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算本轮验证的平均损失和准确率
        val_loss /= len(val_loader)
        val_accuracy = correct_val / len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}: Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}")

        # **学习率调度器**
        scheduler.step(val_loss)  # 根据验证损失调整学习率
        print(f"Current learning rate from scheduler: {scheduler.get_last_lr()}")

        # **分类报告和混淆矩阵**
        print(f"\nClassification Report (Epoch {epoch + 1}):")
        # 输出本轮的分类报告
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

        # 记录损失和准确率
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_accuracy)
        val_acc_list.append(val_accuracy)

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list

# 调用训练与验证函数
train_loss_list, val_loss_list, train_acc_list, val_acc_list = train_and_validate(
    model, train_loader, val_loader, epochs, criterion, optimizer, scheduler, device
)

# **5. 模型保存**
torch.save(model.state_dict(), "food_classifier_optimized.pth")
print("Training complete. Model saved as 'food_classifier_optimized.pth'.")

# **6. 结果可视化**
# 损失和准确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_loss_list, label='Train Loss')
plt.plot(range(1, epochs + 1), val_loss_list, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_acc_list, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_acc_list, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.show()

