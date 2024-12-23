import torchvision
import torchvision.transforms as transforms

# 下载和加载 CIFAR-10 数据集
transform = transforms.Compose(
    [transforms.ToTensor(),  # 转换为Tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 数据标准化
)

# 训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)