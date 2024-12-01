import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import os
from AI_Project2_MyNet import CustomCNN
from AI_Project2_ResNet import SimplifiedResNet
from torchvision.models import resnet18, ResNet18_Weights

# 设置预训练模型的缓存路径
os.environ['TORCH_HOME'] = 'D:/torch_cache'

# **1. 定义 Grad-CAM 类**
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册 forward 和 backward 的 hook 函数
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        """捕获目标层的前向传播输出（特征图）。"""
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        """捕获目标层的反向传播梯度。"""
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        """生成 Grad-CAM 热力图。"""
        # 前向传播
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)  # 如果未指定类别，自动选择预测值最大的类
        score = output[:, class_idx]
        # 反向传播
        self.model.zero_grad()
        score.backward(retain_graph=True)
        # 计算 Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # 全局平均池化梯度
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # 加权和
        cam = F.relu(cam)  # ReLU 激活
        cam = cam.squeeze().cpu().detach().numpy()  # 转换为 numpy 格式
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化
        return cam


# **2. 加载模型**

# 训练好的自定义模型路径
custom_model_path = "./custom_food_classifier.pth"
custom_model = CustomCNN()  # 自定义模型
custom_model.load_state_dict(torch.load(custom_model_path)) # 加载模型参数
custom_model.eval()

# 训练好的简化ResNet模型路径
resnet_model_path = "./food_classifier_optimized.pth"
resnet_model = SimplifiedResNet()  # 自定义模型
resnet_model.load_state_dict(torch.load(resnet_model_path))
resnet_model.eval()

# 预训练的 ResNet 模型
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# 指定目标层（通常是最后一个卷积层）
target_layer = model.layer4[1].conv2

# 创建 Grad-CAM 实例
grad_cam = GradCAM(model, target_layer)

# **3. 数据处理**
# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图像
# 图像路径示例：
# 1. Apple Pie
#image_path = "./food_data/apple_pie/64846.jpg"

# 2. Beet Salad
#image_path = "./food_data/beet_salad/8508.jpg"

# 3. Bread Pudding
#image_path = "./food_data/bread_pudding/19259.jpg"

# 4. Chicken Wings
#image_path = "./food_data/chicken_wings/205980.jpg"

# 5. Dumplings
#image_path = "./food_data/dumplings/10436.jpg"

# 6. Eggs Benedict
#image_path = "./food_data/eggs_benedict/8349.jpg"

# 7. Fish and Chips
#image_path = "./food_data/fish_and_chips/19498.jpg"

# 8. Fried Rice
#image_path = "./food_data/fried_rice/68708.jpg"

# 9. Ice Cream
#image_path = "./food_data/ice_cream/3908953.jpg"

# 10. Pizza
image_path = "./food_data/pizza/329302.jpg"

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # 增加 batch 维度

# **4. 生成 Grad-CAM 热力图**
class_idx = None  # 设置为 None，自动选择预测类
cam = grad_cam.generate(input_tensor, class_idx)

# **5. 可视化**
# 将热力图调整到原始图像大小
heatmap = cv2.resize(cam, (image.size[0], image.size[1]))
heatmap = np.uint8(255 * heatmap)  # 转换为 0-255
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  
## cv2 是 BGR 格式，转换为 RGB 显示，否则红蓝相反
heatmap = heatmap[:, :, ::-1]

# 合并热力图和原始图像
superimposed_img = cv2.addWeighted(np.array(image), 0.7, heatmap, 0.6, 0)

# 显示图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Heatmap")
plt.imshow(heatmap)  
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Superimposed")
# cv2 转换为 RGB 显示
plt.imshow(superimposed_img.astype('uint8'))  
plt.axis("off")

plt.show()
