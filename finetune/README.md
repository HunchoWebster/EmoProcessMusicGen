# MusicGen 模型微调

这个项目提供了用于微调 Meta 的 MusicGen 模型的代码。MusicGen 是一个强大的文本到音乐生成模型，可以通过微调来适应特定的音乐风格或领域。

## 环境要求

- Python 3.8+
- CUDA 支持的 GPU（推荐）
- 足够的磁盘空间用于存储模型和数据集

## 安装

1. 克隆仓库：
```bash
git clone <repository_url>
cd musicgen-finetune
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

1. 准备训练数据：
   - 创建一个包含音频文件的目录
   - 创建一个 JSON 格式的元数据文件，包含以下字段：
     ```json
     [
       {
         "audio_file": "path/to/audio.wav",
         "description": "音频描述文本"
       }
     ]
     ```

2. 准备验证数据（可选）：
   - 与训练数据格式相同
   - 建议使用不同的数据集

## 使用方法

1. 基本训练：
```bash
python train.py \
    --train_data_dir /path/to/train/audio \
    --train_metadata /path/to/train/metadata.json \
    --batch_size 4 \
    --num_epochs 10
```

2. 完整训练（包含验证）：
```bash
python train.py \
    --model_name facebook/musicgen-small \
    --train_data_dir /path/to/train/audio \
    --train_metadata /path/to/train/metadata.json \
    --val_data_dir /path/to/val/audio \
    --val_metadata /path/to/val/metadata.json \
    --batch_size 4 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --save_dir checkpoints
```

## 参数说明

- `--model_name`: 预训练模型名称（默认：facebook/musicgen-small）
- `--train_data_dir`: 训练数据目录（必需）
- `--train_metadata`: 训练数据元数据文件（必需）
- `--val_data_dir`: 验证数据目录（可选）
- `--val_metadata`: 验证数据元数据文件（可选）
- `--batch_size`: 批次大小（默认：4）
- `--num_epochs`: 训练轮数（默认：10）
- `--learning_rate`: 学习率（默认：1e-5）
- `--gradient_accumulation_steps`: 梯度累积步数（默认：1）
- `--save_dir`: 模型保存目录（默认：checkpoints）
- `--device`: 训练设备（默认：cuda 如果可用，否则 cpu）

## 训练监控

训练过程使用 Weights & Biases (wandb) 进行监控。首次运行时需要登录：

```bash
wandb login
```

## 注意事项

1. 确保有足够的 GPU 内存
2. 根据实际需求调整批次大小和学习率
3. 使用梯度累积来处理大批次
4. 定期保存检查点以防训练中断

## 许可证

MIT License 