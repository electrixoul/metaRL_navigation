# Evolution Strategy Training - Minimal System

这是一个最小化的独立系统，用于进化策略（ES）训练网格导航智能体。

## 文件清单（Clean Code重构版）

- `train_evolution_strategy.py` - 主训练脚本（Clean Code重构版）
- `grid_environment.py` - 网格环境实现
- `agent_models.py` - RNN智能体模型（包含12种不同架构）
- `maze_generator.py` - 迷宫生成工具
- `model_checkpoint.py` - 模型保存/加载工具
- `training_logs/` - 训练日志和模型保存目录

## 环境要求

- Python 3.x
- JAX (with CUDA support)
- Flax
- Optax
- NumPy
- OpenCV (cv2) - 用于可视化

## 运行方法

### 1. 退出 conda 环境（重要！）

```bash
conda deactivate
```

### 2. 快速启动（推荐）

```bash
cd minimal_system
python3 train_evolution_strategy.py --strategy GRU --max_generations 100
```

### 3. 自定义参数运行

```bash
python3 train_evolution_strategy.py \
  --population_size 12000 \
  --maze_count 20 \
  --maze_size 10 \
  --neural_network_size 128 \
  --strategy GRU \
  --max_generations 20000 \
  --gpu_id 0
```

## 主要参数说明

- `--population_size`: 种群大小（默认：12000）
- `--maze_count`: 迷宫数量（默认：20）
- `--maze_size`: 迷宫尺寸（默认：10）
- `--neural_network_size`: RNN隐藏层维度（默认：128）
- `--strategy`: RNN架构类型（默认：GRU）
  - 可选：RNN, RNN_th, RNN3, RNN3_th, RNN3_lr, GRU, RNN_sg 等
- `--max_generations`: 最大训练代数（默认：20000）
- `--gpu_id`: GPU设备ID（0-7）
- `--learning_rate`: 学习率（默认：0.01）
- `--sigma`: 初始噪声标准差（默认：0.04）
- `--mean_reward_weight`, `--min_reward_weight`, `--skill_improvement_weight`: 适应度函数权重

## 输出说明

训练过程会每10代输出一次评估结果：
```
[代数, [各迷宫成功率], 最小成功率, 运行时间, 技能提升]
```

模型会自动保存在 `training_logs/` 目录中。

## 性能

- 支持8卡A100并行训练
- 目标性能：1000万+ FPS（每秒环境步数）
- 典型配置下约5秒/代（12000个体 × 20迷宫 × 100步）

## 注意事项

1. **必须退出conda环境**，使用系统原生Python运行
2. 确保 `training_logs/` 目录存在且有写权限
3. 首次运行时JAX会进行JIT编译，可能较慢
4. 程序运行时会占用大量GPU内存
