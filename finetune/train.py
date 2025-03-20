import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from dataset import MusicGenDataset
from tqdm import tqdm
import wandb
from typing import Optional, Dict, Any
import argparse

def setup_wandb(config: Dict[str, Any]) -> None:
    """设置wandb日志记录"""
    wandb.init(
        project="musicgen-finetune",
        config=config,
        name=f"musicgen-{config['model_name']}-{config['batch_size']}bs"
    )

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    processor: Any,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    num_epochs: int,
    device: torch.device,
    save_dir: str,
    gradient_accumulation_steps: int = 1
) -> None:
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        processor: 音频处理器
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        device: 训练设备
        save_dir: 模型保存目录
        gradient_accumulation_steps: 梯度累积步数
    """
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (audio, text) in enumerate(progress_bar):
            # 将数据移到设备
            audio = audio.to(device)
            
            # 处理文本
            inputs = processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # 前向传播
            outputs = model(
                input_values=audio,
                **inputs
            )
            
            # 计算损失
            loss = outputs.loss / gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # 更新学习率
                if scheduler:
                    scheduler.step()
            
            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
            
            # 记录到wandb
            wandb.log({
                'train_loss': loss.item() * gradient_accumulation_steps,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        # 验证
        if val_loader:
            val_loss = validate(model, val_loader, processor, device)
            print(f'Validation Loss: {val_loss:.4f}')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, processor, save_dir, epoch, val_loss)
        else:
            # 保存最新模型
            save_model(model, processor, save_dir, epoch, avg_loss)
        
        # 记录到wandb
        wandb.log({
            'epoch': epoch,
            'epoch_loss': avg_loss,
            'best_val_loss': best_val_loss if val_loader else avg_loss
        })

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    processor: Any,
    device: torch.device
) -> float:
    """
    验证模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        processor: 音频处理器
        device: 验证设备
        
    Returns:
        float: 验证损失
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for audio, text in val_loader:
            audio = audio.to(device)
            
            inputs = processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            outputs = model(
                input_values=audio,
                **inputs
            )
            
            total_loss += outputs.loss.item()
    
    model.train()
    return total_loss / len(val_loader)

def save_model(
    model: nn.Module,
    processor: Any,
    save_dir: str,
    epoch: int,
    loss: float
) -> None:
    """
    保存模型
    
    Args:
        model: 模型
        processor: 音频处理器
        save_dir: 保存目录
        epoch: 当前轮数
        loss: 损失值
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, f'model_epoch_{epoch}_loss_{loss:.4f}')
    model.save_pretrained(model_path)
    
    # 保存处理器
    processor_path = os.path.join(save_dir, f'processor_epoch_{epoch}_loss_{loss:.4f}')
    processor.save_pretrained(processor_path)

def main():
    parser = argparse.ArgumentParser(description='MusicGen模型微调')
    parser.add_argument('--model_name', type=str, default='facebook/musicgen-small',
                      help='预训练模型名称')
    parser.add_argument('--train_data_dir', type=str, required=True,
                      help='训练数据目录')
    parser.add_argument('--train_metadata', type=str, required=True,
                      help='训练数据元数据文件')
    parser.add_argument('--val_data_dir', type=str,
                      help='验证数据目录')
    parser.add_argument('--val_metadata', type=str,
                      help='验证数据元数据文件')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                      help='学习率')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help='梯度累积步数')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='训练设备')
    
    args = parser.parse_args()
    
    # 配置
    config = vars(args)
    
    # 设置wandb
    setup_wandb(config)
    
    # 加载模型和处理器
    model = MusicgenForConditionalGeneration.from_pretrained(args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    # 准备数据集
    train_dataset = MusicGenDataset(
        data_dir=args.train_data_dir,
        metadata_file=args.train_metadata
    )
    
    val_dataset = None
    if args.val_data_dir and args.val_metadata:
        val_dataset = MusicGenDataset(
            data_dir=args.val_data_dir,
            metadata_file=args.val_metadata
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
    
    # 将模型移到设备
    model = model.to(args.device)
    
    # 设置优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.1
    )
    
    # 训练模型
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.save_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # 关闭wandb
    wandb.finish()

if __name__ == '__main__':
    main() 