# 代码重构命名对照表 (Clean Code Standards)

## 重构原则
1. 使用完整、描述性的名称，避免缩写
2. 避免单字母变量名
3. 使用有意义的名称揭示意图
4. 修正拼写错误
5. 遵循Python命名约定（snake_case）

## 全局变量/常量

| 原名称 | 新名称 | 说明 |
|--------|--------|------|
| `es_config` | `evolution_strategy_config` | Evolution Strategy配置 |
| `GE` | `grid_environment` | 网格环境实例 |
| `gpu_list` | `available_gpu_ids` | 可用GPU ID列表 |

## 类名

| 原名称 | 新名称 | 说明 |
|--------|--------|------|
| `ESConfig` | `EvolutionStrategyConfig` | 保持大写驼峰 |
| `GridEnv` | `GridEnvironment` | 环境类名 |

## 配置参数

| 原名称 | 新名称 | 说明 |
|--------|--------|------|
| `pop_size` | `population_size` | 种群大小 |
| `num_landscapes` | `landscape_count` | 地形数量 |
| `num_mazes` | `maze_count` | 迷宫数量 |
| `num_lidar_bins` | `lidar_bin_count` | 激光雷达bin数量 |
| `num_envs` | `environment_count` | 环境数量 |
| `sigma_low` | `sigma_minimum` | 最小sigma值 |
| `sigma_anneal` | `sigma_annealing_rate` | sigma退火率 |
| `use_anealing` | `use_annealing` | 修正拼写 |
| `lr` | `learning_rate` | 学习率 |
| `nn_size` | `neural_network_size` | 神经网络大小 |
| `evo_mode` | `evolution_mode` | 进化模式 |
| `model_pth` | `model_path` | 模型路径 |
| `k1` | `mean_reward_weight` | 平均奖励权重 |
| `k2` | `min_reward_weight` | 最小奖励权重 |
| `k3` | `skill_improvement_weight` | 技能提升权重 |
| `MLP_layers` | `mlp_layer_count` | MLP层数 |

## 函数名

| 原名称 | 新名称 | 说明 |
|--------|--------|------|
| `reorg_obs` | `reorganize_observations` | 重组观测 |
| `get_noise_for_model` | `generate_model_noise` | 生成模型噪声 |
| `init_fn` | `initialize_model` | 初始化模型 |
| `init_fn_rnd` | `initialize_model_random` | 随机初始化 |
| `params_add` | `add_noise_to_parameters` | 参数加噪声 |
| `get_action_deterministic` | `select_deterministic_action` | 确定性动作选择 |
| `get_rnd_act` | `generate_random_actions` | 生成随机动作 |
| `get_fitness_multy_objective` | `calculate_multi_objective_fitness` | 多目标适应度 |
| `get_fitness_final_perf` | `calculate_final_performance_fitness` | 最终表现适应度 |

## 局部变量（主训练循环）

| 原名称 | 新名称 | 说明 |
|--------|--------|------|
| `gen` | `generation` | 代数 |
| `t` | `time_step` | 时间步 |
| `obs` | `observation` | 观测 |
| `concat_obs` | `concatenated_observations` | 拼接观测 |
| `rnn_states` | `recurrent_network_states` | RNN状态 |
| `sr` | `success_rate` | 成功率 |
| `min_sr` | `minimum_success_rate` | 最小成功率 |
| `optim` | `optimizer` | 优化器 |
| `optim_state` | `optimizer_state` | 优化器状态 |
| `pop_noise` | `population_noise` | 种群噪声 |
| `pop_params` | `population_parameters` | 种群参数 |
| `param_center` | `center_parameters` | 中心参数 |
| `y1` | `network_output` | 网络输出 |
| `key_` | `random_key` | 随机数生成器密钥 |

## 批处理变量

| 原名称 | 新名称 | 说明 |
|--------|--------|------|
| `batched_actions` | `batch_actions` | 批量动作 |
| `batched_goal_reached` | `batch_goals_reached` | 批量目标到达 |
| `batched_episode_reward` | `batch_episode_rewards` | 批量回合奖励 |
| `batched_task_duration` | `batch_task_durations` | 批量任务持续时间 |
| `batched_task_steps` | `batch_task_steps` | 批量任务步数 |
| `batched_skill_improvement` | `batch_skill_improvements` | 批量技能提升 |
| `batched_first_task_duration` | `batch_first_task_durations` | 批量首次任务时长 |
| `batched_first_task_tag` | `batch_first_task_flags` | 批量首次任务标记 |
| `batched_task_final_perf` | `batch_final_performances` | 批量最终表现 |

## 特殊参数

| 原名称 | 新名称 | 说明 |
|--------|--------|------|
| `meditation` | `action_threshold` | 动作选择阈值 |
| `demonstration` | `minimum_performance_threshold` | 最小性能阈值 |

## vmap函数

| 原名称 | 新名称 | 说明 |
|--------|--------|------|
| `model_forward_vmap` | `vectorized_model_forward` | 向量化前向传播 |
| `get_action_deterministic_vmap` | `vectorized_select_deterministic_action` | 向量化动作选择 |
| `get_fitness_multy_objective_vmap` | `vectorized_calculate_multi_objective_fitness` | 向量化多目标适应度 |
| `get_fitness_final_perf_vmap` | `vectorized_calculate_final_performance_fitness` | 向量化最终表现适应度 |

## 时间相关变量

| 原名称 | 新名称 | 说明 |
|--------|--------|------|
| `start_time` | `training_start_time` | 训练开始时间 |
| `end_time` | `evaluation_end_time` | 评估结束时间 |
| `eval_start` | `evaluation_start_time` | 评估开始时间 |
| `reset_start` | `reset_start_time` | 重置开始时间 |
| `step_start` | `step_start_time` | 步骤开始时间 |
| `opt_start` | `optimization_start_time` | 优化开始时间 |
| `inference_start` | `inference_start_time` | 推理开始时间 |
| `time_` | `elapsed_time` | 消耗时间 |

---

**重构状态**: 🔄 进行中
**最后更新**: 2025-11-10
