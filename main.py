import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import scipy.io.arff as arff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# 尝试导入 AME 模块，如果未找到则提示错误
try:
    from AME import AttentionFusion
except ImportError:
    raise ImportError("请确保 'AME.py' 文件存在于当前目录，并且定义了 'AttentionFusion' 类。")

# 配置和超参数
TRAINSET_FILE = 'dataset/ECG5000_TRAIN.arff'
TESTSET_FILE = 'dataset/ECG5000_TEST.arff'
SEQ_LEN = 140
N_FEATURES = 1
BATCH_SIZE = 512
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
HIDDEN_SIZE = 128
NUM_LAYERS = 3
NUM_CLASSES = 2
LEARNING_RATE = 0.0005
NUM_EPOCHS = 40
PLOT_DIR = 'plots'

# 创建保存混淆矩阵的目录
os.makedirs(PLOT_DIR, exist_ok=True)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'使用设备: {device}')

# 定义 ECG5000 数据集
class ECG5000Dataset(Dataset):
    def __init__(self, train_file, test_file):
        # 加载数据
        traindata, trainmeta = arff.loadarff(train_file)
        testdata, testmeta = arff.loadarff(test_file)
        train = pd.DataFrame(traindata, columns=trainmeta.names())
        test = pd.DataFrame(testdata, columns=testmeta.names())
        
        # 合并训练和测试数据
        df = pd.concat([train, test]).reset_index(drop=True)

        # 重命名最后一列为 'target'
        new_columns = list(df.columns)
        new_columns[-1] = 'target'
        df.columns = new_columns

        # 转换目标为整数并进行二分类
        df['target'] = df['target'].apply(lambda x: int(x.decode('utf-8')) - 1)
        df['target'] = (df['target'] != 0).astype(int)  # 0=normal, 1=abnormal

        # 存储标签和特征
        self.labels = df['target'].values
        self.X = df.drop(labels='target', axis=1).astype(np.float32).to_numpy()

    def __getitem__(self, index):
        data = torch.from_numpy(self.X[index]).reshape(-1, 1)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return data, label

    def __len__(self):
        return self.X.shape[0]

# 创建数据集
dataset = ECG5000Dataset(TRAINSET_FILE, TESTSET_FILE)

# 打印数据集信息
print(f'总样本数: {len(dataset)}')
print(f'类别分布: {np.bincount(dataset.labels)}')

# 数据集划分
def create_data_loaders(dataset, validation_split, test_split, batch_size, random_seed=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_valid = int(np.floor(validation_split * dataset_size))
    split_test = int(np.floor(test_split * dataset_size))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    test_indices = indices[:split_test]
    val_indices = indices[split_test:split_test + split_valid]
    train_indices = indices[split_test + split_valid:]
    
    # 创建采样器
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, validation_loader, test_loader

train_loader, validation_loader, test_loader = create_data_loaders(
    dataset, VALIDATION_SPLIT, TEST_SPLIT, BATCH_SIZE, RANDOM_SEED
)

# 定义包含 AME 的 LSTM 注意力分类器
class LSTMAttentionClassifierWithAME(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, seq_len):
        super(LSTMAttentionClassifierWithAME, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.ame = AttentionFusion(seq_len, hidden_size * 2, num_layers * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        # LSTM 输出
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # AME 模块增强
        mid_out = self.ame(lstm_out)
        
        # 注意力机制
        attention_weights = torch.softmax(self.attention(mid_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 分类
        out = self.fc(context_vector)
        return out

# 定义不包含 AME 的 LSTM 注意力分类器
class LSTMAttentionClassifierWithoutAME(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMAttentionClassifierWithoutAME, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        # LSTM 输出
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 注意力机制
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 分类
        out = self.fc(context_vector)
        return out

# 训练函数
def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs, model_name):
    model.train()
    for epoch in tqdm(range(num_epochs), desc=f"Training {model_name}"):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/(i+1):.4f}, Acc: {100 * correct / total:.2f}%')
                
    print(f'\n完成训练 {model_name}。\n')

# 测试函数
def test_model(model, test_loader, model_name):
    model.eval()
    all_labels = []
    all_preds = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'\nTest Accuracy ({model_name}): {accuracy:.2f}%')
    
    # 生成混淆矩阵和分类报告
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    print(f'\nConfusion Matrix ({model_name}):')
    print(cm)
    
    print(f'\nClassification Report ({model_name}):')
    print(report)
    
    return cm, report, all_labels, all_preds

# 绘制并保存混淆矩阵
def plot_confusion_matrix(cm, model_name, plot_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix ({model_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'已保存混淆矩阵图: {plot_path}\n')

# 主流程
if __name__ == "__main__":
    # 定义超参数
    input_size = N_FEATURES
    hidden_size = HIDDEN_SIZE
    num_layers = NUM_LAYERS
    num_classes = NUM_CLASSES
    learning_rate = LEARNING_RATE
    num_epochs = NUM_EPOCHS
    
    # 初始化包含 AME 的模型
    model_with_ame = LSTMAttentionClassifierWithAME(
        input_size, hidden_size, num_layers, num_classes, SEQ_LEN
    ).to(device)
    
    # 初始化不包含 AME 的模型
    model_without_ame = LSTMAttentionClassifierWithoutAME(
        input_size, hidden_size, num_layers, num_classes
    ).to(device)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer_with_ame = torch.optim.Adam(model_with_ame.parameters(), lr=learning_rate)
    optimizer_without_ame = torch.optim.Adam(model_without_ame.parameters(), lr=learning_rate)
    
    # 启用异常检测（调试阶段使用，训练完成后可关闭以提升性能）
    torch.autograd.set_detect_anomaly(True)
    
    # 训练包含 AME 的模型
    train_model(model_with_ame, train_loader, validation_loader, 
                criterion, optimizer_with_ame, num_epochs, "With_AME")
    
    # 测试包含 AME 的模型
    cm_with_ame, report_with_ame, _, _ = test_model(model_with_ame, test_loader, "With_AME")
    plot_confusion_matrix(cm_with_ame, "With_AME", PLOT_DIR)
    
    # 训练不包含 AME 的模型
    train_model(model_without_ame, train_loader, validation_loader, 
                criterion, optimizer_without_ame, num_epochs, "Without_AME")
    
    # 测试不包含 AME 的模型
    cm_without_ame, report_without_ame, _, _ = test_model(model_without_ame, test_loader, "Without_AME")
    plot_confusion_matrix(cm_without_ame, "Without_AME", PLOT_DIR)
    
    print("所有模型训练和评估完成。")
