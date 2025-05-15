# ResNet-Transformer 心电分类项目

## 项目概述
融合ResNet特征提取与Transformer时序建模的心电信号分类模型，实现对11种MI和HC的分类识别。

## 环境要求
- Python 3.7+
- CUDA 11.3+
- PyTorch 1.7+

## 安装依赖
```bash 
pip install -r requirements.txt
```

## 数据准备
1. 原始数据存放于`ST image`目录
2. 按疾病类型分12个子目录（例如：Healthy control (no)/）
3. 每个子目录存放S变换图像（示例：heartbeat_1.png）
4. 预处理流程包含：
   - 图像尺寸统一调整为224x224
   - 进行标准化处理

## 训练与测试
```bash
# 启动训练
python main.py

# 自定义参数示例
python main.py \
    --data_dir ST_image \
    --batch_size 64 \
    --num_epochs 60 \
    --save_dir my_model
```

## 代码文件说明

| 文件 | 功能描述 |
|------|----------|
| `main.py` | 主入口脚本，负责参数解析、训练流程控制及模型保存 |
| `model.py` | 定义ResNet-Transformer混合模型架构，包含特征融合模块和梯度激活处理 |
| `dataset.py` | 实现ST image数据加载 |
| `gradcam.py` | 提供Grad-CAM可视化功能，生成模型注意力热力图 |

## 参数说明
以下是模型的训练相关参数，可以在main.py中进行修改，默认参数及说明如下：

| 参数 | 默认值 | 说明 |
|------|--------|-----|
| --data_dir | ST image | 数据目录路径 |
| --batch_size | 64 | 训练批次大小 |
| --num_epochs | 60 | 训练总轮数 |
| --lr | 0.001 | 初始学习率 |
| --num_classes | 12 | 分类类别数量 |
| --seed | 42 | 随机种子 |
| --save_dir | best_model | 模型保存目录 |
| --result_dir | results | 训练结果目录 |


## Grad-CAM可视化
1. 训练完成后使用以下脚本生成热力图分析
2. 结果保存在`gradcam_results`目录
3. 使用示例：

```python
from utils.gradcam import GradCAM
# 加载训练好的模型
gradcam = GradCAM(model.module, model.module.resnet.layer4)
heatmap = gradcam.generate(input_tensor)
heatmap.save('results/gradcam/sample_heatmap.png')
```

## 模型架构
<img src="Figure/RTCF.png" width="600" />

- ResNet50骨干网络提取空间特征
- Transformer编码器捕捉时序依赖
- 全连接层后通过Softmax输出分类结果