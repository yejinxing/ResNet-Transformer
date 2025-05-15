import torch
import torch.nn as nn
import cv2
import numpy as np
from matplotlib import pyplot as plt
from dataset import create_loaders
from model import ResNetTransformer
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from gradcam import GradCAM

# 训练配置
config = {
    'result_dir': 'results',
    'data_dir': 'ST image',
    'batch_size': 64,
    'num_epochs': 60,
    'lr': 0.001,
    'num_classes': 12,
    'save_dir': 'best_model',
    'seed': 42
}

def test_model(model_path, num_classes, result_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据加载器
    _, _, test_loader = create_loaders(
        config['data_dir'], 
        config['batch_size'],
        seed=config['seed']
    )
    
    # 加载模型
    model = ResNetTransformer(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # 初始化指标
    test_metrics = {
        'test_loss': [],
        'test_acc': [],
        'all_labels': [],
        'all_preds': []
    }
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        total = 0
        correct = 0
        test_loss = 0.0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_metrics['all_labels'].extend(labels.cpu().numpy())
            test_metrics['all_preds'].extend(predicted.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_metrics['test_loss'] = test_loss / len(test_loader.dataset)
    test_metrics['test_acc'] = correct / total
    
    # 保存测试指标
    np.save(os.path.join(result_dir, 'test_metrics.npy'), test_metrics)
    return test_metrics


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['save_dir'], exist_ok=True)

    # 数据加载配置
    config['seed'] = 42  # 添加随机种子配置
    
    # 创建数据加载器（修复缩进问题）
    train_loader, val_loader, test_loader = create_loaders(
        config['data_dir'], 
        config['batch_size'],
        seed=config['seed']
    )

    # 模型初始化
    model = ResNetTransformer(num_classes=config['num_classes'])
    model = model.to(device)

    # 优化器和损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3)
    criterion = nn.CrossEntropyLoss()

        # 初始化指标记录
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'all_labels': [],
        'all_preds': []
    }
    os.makedirs(config['result_dir'], exist_ok=True)

    best_acc = 0.0
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                metrics['all_labels'].extend(labels.cpu().numpy())
                metrics['all_preds'].extend(predicted.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 统计指标
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        # 学习率调整
        scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))

        print(f'Epoch {epoch+1}/{config["num_epochs"]}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # 记录指标
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
    # 训练后执行Grad-CAM分析
    class_names = ['HC','AMI','ALMI','ASMI','ASLMI','IMI','ILMI','IPMI','IPLMI','LMI','PMI','PLMI']  # 实际心脏病分类名称
    gradcam = GradCAM(model, model.target_layer)
    sample_inputs, _ = next(iter(val_loader))
    sample_inputs = sample_inputs.to(device)
    
    # 生成并保存所有类别热力图
    plt = gradcam.generate_all_classes(sample_inputs, class_names)
    os.makedirs('gradcam_results', exist_ok=True)
    plt.savefig('gradcam_results/all_classes_heatmap.png')
    plt.close()

def plot_metrics(metrics, result_dir):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['val_acc'], label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_metrics.png'))
    plt.close()


def plot_confusion_matrix(labels, preds, num_classes, result_dir, class_names):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
    plt.close()


if __name__ == '__main__':
    train_metrics = train()
    plot_metrics(train_metrics, config['result_dir'])
    plot_confusion_matrix(
        train_metrics['all_labels'],
        train_metrics['all_preds'],
        config['num_classes'],
        config['result_dir'],
        train_metrics['class_names']
    )