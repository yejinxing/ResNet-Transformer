import argparse
import os
import cv2
import numpy as np
import torch
from dataset import create_loaders
from model import ResNetTransformer
from train import train, plot_metrics, plot_confusion_matrix, test_model
from gradcam import GradCAM

# 参数配置
def parse_args():
    parser = argparse.ArgumentParser(description='ResNet-Transformer 训练脚本')
    parser.add_argument('--data_dir', default='ST image', help='数据集目录路径')
    parser.add_argument('--batch_size', type=int, default=64, help='训练批次大小')
    parser.add_argument('--num_epochs', type=int, default=60, help='训练总周期数')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--save_dir', default='best_model', help='模型保存目录')
    parser.add_argument('--result_dir', default='results', help='训练结果目录')
    parser.add_argument('--num_classes', type=int, default=12, help='分类类别数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()

# 主执行流程
def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    metrics = train()
    
    plot_metrics(metrics, args.result_dir)
    plot_confusion_matrix(
        metrics['all_labels'],
        metrics['all_preds'],
        args.num_classes,
        args.result_dir
    )

   
    best_model = ResNetTransformer(num_classes=args.num_classes)
    best_model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    
    # 获取验证集数据加载器
    _, val_loader, _ = create_loaders(args.data_dir, args.batch_size)
    sample_inputs, _ = next(iter(val_loader))
    
    # 生成并保存热力图
    gradcam = GradCAM(best_model, best_model.target_layer)
    heatmap = gradcam.get_heatmap(sample_inputs)
    img = sample_inputs[0].numpy().transpose(1, 2, 0)
    img = (img * 255).astype('uint8')
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.5, heatmap, 0.4, 0)
    os.makedirs('gradcam_results', exist_ok=True)
    cv2.imwrite('gradcam_results/final_analysis.jpg', superimposed_img)
    
    run_test_phase(
        model_path=os.path.join(args.save_dir, 'best_model.pth'),
        num_classes=args.num_classes,
        result_dir=os.path.join(args.result_dir, 'test')
    )

# 测试
def run_test_phase(model_path, num_classes, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    
    # 加载模型并测试
    test_metrics = test_model(
        model_path,
        num_classes,
        result_dir
    )
    
    # 保存测试结果
    plot_metrics(test_metrics, result_dir)
    plot_confusion_matrix(
        test_metrics['all_labels'],
        test_metrics['all_preds'],
        num_classes,
        result_dir
    )
    return test_metrics

if __name__ == '__main__':
    main()
    run_test_phase()