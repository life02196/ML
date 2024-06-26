import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision.models import densenet201, DenseNet201_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用於進度條
from datetime import datetime  # 用於獲取當前時間
import inspect  # 用於獲取代碼
import os  # 用於文件操作

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用 DenseNet201 的預訓練權重
weights = DenseNet201_Weights.IMAGENET1K_V1

# 定義 CIFAR-10 數據集的轉換，根據 DenseNet201 的預處理要求
transform = transforms.Compose([
    transforms.Resize(weights.transforms().resize_size),
    transforms.CenterCrop(weights.transforms().crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

# 加載 CIFAR-10 訓練和測試數據集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 縮減訓練和測試數據集
train_size = 50  # 訓練集大小
test_size = 10   # 測試集大小
train_subset, _ = random_split(trainset, [train_size, len(trainset) - train_size])
test_subset, _ = random_split(testset, [test_size, len(testset) - test_size])

batch_size = 32
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 加載預訓練的DenseNet201模型並使用預訓練權重
model = densenet201(weights=weights)
model.classifier = nn.Linear(model.classifier.in_features, 10)  # 修改輸出層以適應CIFAR-10 (10類)

model = model.to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)  # 用於 early stopping

# 訓練和測試記錄
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
num_epoch = 200
patience_counter = 0  # 初始化 patience counter

# 定義保存和繪圖的函數
def save_and_plot():
    final_accuracy = test_accuracies[-1] if test_accuracies else 0
    epochs = range(0, len(train_losses))
    print(f'Final Test Accuracy: {final_accuracy:.2f}%')
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epochs')

    if test_accuracies:
        plt.annotate(f'Final Test Accuracy: {final_accuracy:.2f}%', 
                     xy=(epochs[-1], test_accuracies[-1]), 
                     xytext=(epochs[-1], test_accuracies[-1] + 5),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = 'densenet201'
    dataset_name = 'CIFAR10'
    plt.savefig(f'{model_name}_{dataset_name}_batch{batch_size}_training_results_{current_time}.png')
    plt.show()

    torch.save(model.state_dict(), f'{model_name}_{dataset_name}_batch{batch_size}_{current_time}.pth')

# 保存當前代碼到文件
def save_current_code():
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = 'densenet201'
    dataset_name = 'CIFAR10'
    filename = f'{model_name}_{dataset_name}_batch{batch_size}_current_code_{current_time}.py'
    with open(filename, 'w') as f:
        f.write(inspect.getsource(inspect.getmodule(save_current_code)))

# 訓練模型
try:
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        if epoch == 0:
            print('train start')

        train_loader_tqdm = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epoch} - Training", unit="batch")
        for i, data in enumerate(train_loader_tqdm):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_loader_tqdm.set_postfix(loss=running_loss / (i + 1))

        # 記錄訓練損失和準確率
        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(100 * correct_train / total_train)

        # 測試模型
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        with torch.no_grad():
            test_loader_tqdm = tqdm(testloader, desc=f"Epoch {epoch+1}/{num_epoch} - Testing", unit="batch")
            for data in test_loader_tqdm:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_losses.append(test_loss / len(testloader))
        test_accuracies.append(100 * correct_test / total_test)

        # 更新學習率
        scheduler.step(test_loss / len(testloader))

        # early stopping 的簡單實現
        if len(test_losses) > 1 and test_losses[-1] > test_losses[-2]:
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= 10:  # 如果測試損失在連續 10 次 epoch 中沒有改善，則停止訓練
            print('Early stopping triggered')
            save_and_plot()
            save_current_code()     
            break

        # 打印當前 epoch 的訓練和測試結果
        print(f'Epoch {epoch+1}/{num_epoch}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.2f}%')

except KeyboardInterrupt:
    print("Training interrupted. Saving current progress...")

finally:
    save_and_plot()
    save_current_code()

print('Finished Training')
