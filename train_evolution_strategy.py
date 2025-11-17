"""Evolution Strategy Training for Grid Navigation with Repeat Tasks"""

import argparse
import os
import time
import numpy as np
import numpy.random as npr
import jax
import jax.numpy as jnp
import jax.flatten_util
from functools import partial
from jax import jit
import optax

from grid_environment import *
from agent_models import *
from model_checkpoint import *
from maze_generator import *

# Constants
AVAILABLE_GPU_IDS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ACTION_COUNT = 4
NO_ACTION_INDEX = 4


class EvolutionStrategyConfig:
    """Configuration for Evolution Strategy training system."""
    # Population and Evolution
    population_size: int = 12000
    landscape_count: int = 1
    sigma: float = 0.04
    sigma_minimum: float = 0.02
    sigma_annealing_rate: float = 0.999
    # Optimization
    learning_rate: float = 0.01
    weight_decay: float = 0.02
    optimizer: str = 'adam'
    # Training
    steps_per_episode: int = 100
    max_generations: int = 20000
    reward_type: str = 'dense'
    evolution_mode: int = 1
    use_annealing: bool = True
    # Environment
    maze_size: int = 10
    maze_count: int = 20
    maze_weight: float = -1.0
    observation_type: str = 'local'
    lidar_bin_count: int = 32
    # Neural Network
    neural_network_size: int = 128
    strategy: str = 'GRU'
    mlp_layer_count: int = 2
    deterministic: str = 'False'
    # Fitness Weights
    mean_reward_weight: float = 1.0
    min_reward_weight: float = 2.0
    skill_improvement_weight: float = 0.0
    # Model and Evaluation
    model_path: str = ''
    minimum_performance_threshold: float = 1.0
    action_threshold: float = -10.0
    # System
    gpu_id: str = '0'
    seed: int = 0
    landscape_id: int = 0


@partial(jax.jit, static_argnames=('landscape_count',))
def reorganize_observations(observation, landscape_count):
    envs_per_landscape = observation.shape[0] // landscape_count
    return jnp.reshape(observation, (envs_per_landscape, landscape_count, observation.shape[1]))


@partial(jax.jit, static_argnums=(2,))
def generate_model_noise(random_key, parameters, population_size, standard_deviation: float = 0.02):
    """Generate mirrored noise for antithetic sampling."""
    half_pop = population_size // 2
    num_vars = len(jax.tree_util.tree_leaves(parameters))
    tree_def = jax.tree_util.tree_structure(parameters)
    all_keys = jax.random.split(random_key, num=num_vars)
    
    noise = jax.tree_util.tree_map(
        lambda p, k: standard_deviation * jax.random.normal(k, shape=(half_pop, *p.shape), dtype=p.dtype),
        parameters, jax.tree_util.tree_unflatten(tree_def, all_keys))
    return jax.tree_util.tree_map(lambda n: jnp.concatenate([n, -n], axis=0), noise)


@partial(jax.jit, static_argnums=(3,))
def forward_pass_model(variables, state, input_data, model):
    return model.apply(variables, state, input_data)

vectorized_model_forward = jax.vmap(forward_pass_model, in_axes=(0, 0, 0, None), out_axes=0)


@partial(jax.jit, static_argnums=(2,))
def initialize_model(random_seed, concatenated_observations, model):
    batch_size = concatenated_observations.shape[0]
    initial_state = model.initial_state(batch_size)
    return model.init(random_seed, initial_state, concatenated_observations)


@partial(jax.jit, donate_argnums=(0, 1))
def add_noise_to_parameters(center_parameters, population_noise):
    return jax.tree_util.tree_map(lambda x, y: x + y, center_parameters, population_noise)


@partial(jax.jit, static_argnums=(1,))
def select_deterministic_action(network_output, action_threshold: float = 0.2):
    max_idx = jnp.argmax(network_output)
    return jnp.where(jnp.max(network_output) >= action_threshold, max_idx, NO_ACTION_INDEX)

vectorized_select_deterministic_action = jax.vmap(select_deterministic_action, in_axes=(0, None))


@partial(jax.jit, static_argnums=(0,))
def generate_random_actions(environment_count):
    return jax.random.randint(jax.random.PRNGKey(npr.randint(0, 1000000)), 
                             shape=(environment_count,), minval=0, maxval=ACTION_COUNT)


@jit
def _centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Centered rank transformation from: https://arxiv.org/pdf/1703.03864.pdf"""
    shape = x.shape
    x = x.ravel()
    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - 0.5
    return x.reshape(shape)


@partial(jax.jit, static_argnums=(2, 3, 4))
def calculate_multi_objective_fitness(rewards, skill_improvement,
                                      mean_weight=1.0, min_weight=2.0, skill_weight=1.0):
    return (mean_weight * jnp.mean(rewards) + min_weight * jnp.min(rewards) + 
            skill_weight * jnp.mean(skill_improvement))

vectorized_calculate_multi_objective_fitness = jax.vmap(
    calculate_multi_objective_fitness, in_axes=(0, 0, None, None, None))


@jit
def calculate_final_performance_fitness(final_performance):
    return jnp.mean(final_performance)

vectorized_calculate_final_performance_fitness = jax.vmap(calculate_final_performance_fitness)


def add_padding_to_landscapes(landscapes, width, height):
    """Add border padding around landscapes."""
    padded = []
    for landscape in landscapes:
        land_2d = np.array(landscape).reshape(height - 2, width - 2)
        padded_land = np.pad(land_2d, 1, 'constant', constant_values=0)
        padded.append(np.reshape(padded_land, (width * height,)))
    return padded


def main():
    """Main training loop."""
    config = EvolutionStrategyConfig()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='ES Training for Grid Navigation')
    parser.add_argument('--population_size', type=int, default=config.population_size)
    parser.add_argument('--sigma', type=float, default=config.sigma)
    parser.add_argument('--sigma_minimum', type=float, default=config.sigma_minimum)
    parser.add_argument('--sigma_annealing_rate', type=float, default=config.sigma_annealing_rate)
    parser.add_argument('--learning_rate', type=float, default=config.learning_rate)
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay)
    parser.add_argument('--steps_per_episode', type=int, default=config.steps_per_episode)
    parser.add_argument('--max_generations', type=int, default=config.max_generations)
    parser.add_argument('--maze_size', type=int, default=config.maze_size)
    parser.add_argument('--maze_count', type=int, default=config.maze_count)
    parser.add_argument('--maze_weight', type=float, default=config.maze_weight)
    parser.add_argument('--neural_network_size', type=int, default=config.neural_network_size)
    parser.add_argument('--strategy', type=str, default=config.strategy)
    parser.add_argument('--mean_reward_weight', type=float, default=config.mean_reward_weight)
    parser.add_argument('--min_reward_weight', type=float, default=config.min_reward_weight)
    parser.add_argument('--skill_improvement_weight', type=float, default=config.skill_improvement_weight)
    parser.add_argument('--model_path', type=str, default=config.model_path)
    parser.add_argument('--minimum_performance_threshold', type=float, default=config.minimum_performance_threshold)
    parser.add_argument('--action_threshold', type=float, default=config.action_threshold)
    parser.add_argument('--evolution_mode', type=int, default=config.evolution_mode)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=config.seed)
    
    args = parser.parse_args()
    
    # Update config
    config.population_size = args.population_size
    config.sigma = args.sigma
    config.sigma_minimum = args.sigma_minimum
    config.sigma_annealing_rate = args.sigma_annealing_rate
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.steps_per_episode = args.steps_per_episode
    config.max_generations = args.max_generations
    config.maze_size = args.maze_size
    config.maze_count = args.maze_count
    config.maze_weight = args.maze_weight
    config.neural_network_size = args.neural_network_size
    config.strategy = args.strategy
    config.mean_reward_weight = args.mean_reward_weight
    config.min_reward_weight = args.min_reward_weight
    config.skill_improvement_weight = args.skill_improvement_weight
    config.model_path = args.model_path
    config.minimum_performance_threshold = args.minimum_performance_threshold
    config.action_threshold = args.action_threshold
    config.evolution_mode = args.evolution_mode
    config.gpu_id = AVAILABLE_GPU_IDS[args.gpu_id]
    config.seed = args.seed
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    
    # Generate landscapes
    maze_w = config.maze_size + 2
    maze_h = config.maze_size + 2
    landscapes = generate_maze_pool(num_mazes=config.maze_count, width=config.maze_size, 
                                   height=config.maze_size, weight=config.maze_weight)
    landscapes = add_padding_to_landscapes(landscapes, width=maze_w, height=maze_h)
    config.landscape_count = count_landscapes(landscapes)
    
    # Create environment
    print("Creating environment...")
    env_start = time.time()
    grid_env = GridEnv(landscapes=landscapes, width=maze_w, height=maze_h,
                      num_envs_per_landscape=config.population_size, empty=False,
                      num_lidar_bins=config.lidar_bin_count)
    print(f"Environment created in {time.time() - env_start:.2f}s")
    
    # Create model
    hidden_size = config.neural_network_size
    model_map = {
        "RNN": RNN, "GRU": lambda h: GRU(in_dims=9, hidden_dims=h),
        "RNN3": RNN3, "RNN4": RNN3_th, "RNN5": RNN3_lr, "RNN6": RNN_th,
        "RNN7": RNN_sg, "RNN8": RNN3_sg, "RNN9": RNN_th2, "RNN10": RNN_th3,
        "RNN11": RNN_th_rs, "RNN12": RNN_th_rs1
    }
    model = model_map[config.strategy](hidden_dims=hidden_size) if config.strategy != "GRU" else model_map[config.strategy](hidden_size)
    
    # Initialize
    random_key = jax.random.PRNGKey(config.seed if config.seed != 0 else npr.randint(0, 1000000000000))
    print(f"Random key: {random_key}")
    
    concat_obs_reorg = reorganize_observations(grid_env.concat_obs, config.landscape_count)
    center_params = (load_weights(config.model_path) if config.model_path 
                    else initialize_model(random_key, concat_obs_reorg[0], model))
    
    pop_noise = generate_model_noise(random_key, center_params, config.population_size, config.sigma)
    pop_params = add_noise_to_parameters(center_params, pop_noise)
    batch_actions = generate_random_actions(grid_env.num_envs)
    
    flat_params, _ = jax.flatten_util.ravel_pytree(center_params)
    print(f"Parameter memory: {flat_params.nbytes / 1024 / 1024:.4f} MB")
    
    # Initialize optimizer
    optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    opt_state = optimizer.init(center_params)
    
    # Training loop
    current_sigma = config.sigma
    best_skill = -100000
    train_start = time.time()
    
    print(f"\nStarting training: k1={config.mean_reward_weight}, k2={config.min_reward_weight}, k3={config.skill_improvement_weight}")
    print(f"Population: {config.population_size}, Generations: {config.max_generations}, Mazes: {config.maze_count}")
    
    for generation in range(config.max_generations):
        # Generate new landscapes
        landscapes = generate_maze_pool(num_mazes=config.maze_count, width=config.maze_size,
                                       height=config.maze_size, weight=config.maze_weight)
        landscapes = add_padding_to_landscapes(landscapes, width=maze_w, height=maze_h)
        grid_env.set_landscapes(landscapes)
        
        # Generate population
        new_key, run_key = jax.random.split(random_key)
        random_key = new_key
        current_sigma = max(current_sigma * config.sigma_annealing_rate, config.sigma_minimum)
        pop_noise = generate_model_noise(run_key, center_params, config.population_size, current_sigma)
        pop_params = add_noise_to_parameters(center_params, pop_noise)
        
        # Evaluation every 10 generations
        if generation % 10 == 0:
            eval_end = time.time()
            grid_env.reset()
            total_envs = config.population_size * config.landscape_count
            rnn_states = model.initial_state(total_envs)
            concat_obs = grid_env.concat_obs
            
            rnn_states, net_out = forward_pass_model(center_params, rnn_states, concat_obs, model)
            batch_actions = vectorized_select_deterministic_action(net_out, config.action_threshold)
            
            batch_rewards = jnp.zeros_like(grid_env.batched_goal_reached, dtype=np.float32)
            batch_durations = jnp.zeros_like(grid_env.batched_goal_reached, dtype=np.float32)
            batch_steps = jnp.zeros_like(grid_env.batched_goal_reached, dtype=np.float32)
            batch_skill = jnp.zeros_like(grid_env.batched_goal_reached, dtype=np.float32)
            batch_first_dur = jnp.zeros_like(grid_env.batched_goal_reached, dtype=np.float32)
            batch_first_flag = jnp.ones_like(grid_env.batched_goal_reached, dtype=np.float32)
            
            for t in range(config.steps_per_episode):
                goals_reached, concat_obs = grid_env.step(batch_actions)
                rnn_states, net_out = forward_pass_model(center_params, rnn_states, concat_obs, model)
                batch_actions = vectorized_select_deterministic_action(net_out, config.action_threshold)
                
                batch_rewards = jnp.where(goals_reached, batch_rewards + 1, batch_rewards)
                batch_durations = jnp.where(goals_reached, t - batch_steps, batch_durations)
                batch_steps = jnp.where(goals_reached, t, batch_steps)
                batch_first_dur = jnp.where(jnp.logical_and(batch_first_flag, batch_rewards), t, batch_first_dur)
                batch_first_flag = jnp.where(batch_rewards, 0, batch_first_flag)
                batch_skill = jnp.where(batch_rewards, batch_first_dur - batch_durations, batch_skill)
                if t == config.steps_per_episode - 1:
                    batch_skill = jnp.where(batch_rewards, batch_skill, -config.steps_per_episode)
            
            batch_rewards = jnp.reshape(batch_rewards, (config.population_size, config.landscape_count))
            batch_skill = jnp.divide(batch_skill, config.steps_per_episode)
            success_rates = jnp.mean(batch_rewards, axis=0)
            min_sr = jnp.min(success_rates)
            mean_skill = jnp.mean(batch_skill)
            
            if min_sr >= config.minimum_performance_threshold and mean_skill >= best_skill:
                save_weights(f"./training_logs/rnn_model_idealobs_{generation}_{config.gpu_id}", center_params)
                best_skill = mean_skill
            
            elapsed = eval_end - train_start
            print(f"[{generation}, {[f'{sr:.4f}' for sr in success_rates]}, {min_sr:.4f}, {elapsed:.2f}, {mean_skill:.4f}]")
        
        if generation % 50 == 0 and generation > 0:
            print(f"Gen {generation}: sigma={current_sigma:.6f}, best_skill={best_skill:.4f}")
        
        # Training phase
        grid_env.reset()
        total_envs = config.population_size * config.landscape_count
        rnn_states = model.initial_state(total_envs)
        batch_rnn = jnp.reshape(rnn_states, (grid_env.num_envs // config.landscape_count,
                                             config.landscape_count, rnn_states.shape[-1]))
        
        batch_rewards = jnp.zeros_like(grid_env.batched_goal_reached, dtype=np.float32)
        batch_durations = jnp.zeros_like(grid_env.batched_goal_reached, dtype=np.float32)
        batch_final_perf = jnp.zeros_like(grid_env.batched_goal_reached, dtype=np.float32)
        batch_steps = jnp.zeros_like(grid_env.batched_goal_reached, dtype=np.float32)
        batch_skill = jnp.zeros_like(grid_env.batched_goal_reached, dtype=np.float32)
        batch_first_dur = jnp.zeros_like(grid_env.batched_goal_reached, dtype=np.float32)
        batch_first_flag = jnp.ones_like(grid_env.batched_goal_reached, dtype=np.float32)
        
        for t in range(config.steps_per_episode):
            new_key, run_key = jax.random.split(random_key)
            random_key = new_key
            
            goals_reached, concat_obs = grid_env.step(batch_actions)
            batch_obs = jnp.reshape(concat_obs, (grid_env.num_envs // config.landscape_count,
                                                 config.landscape_count, concat_obs.shape[-1]))
            batch_rnn, net_out = vectorized_model_forward(pop_params, batch_rnn, batch_obs, model)
            net_out_flat = jnp.reshape(net_out, (grid_env.num_envs, net_out.shape[-1]))
            batch_actions = vectorized_select_deterministic_action(net_out_flat, config.action_threshold)
            
            batch_rewards = jnp.where(goals_reached, batch_rewards + 1, batch_rewards)
            batch_durations = jnp.where(goals_reached, t - batch_steps, batch_durations)
            batch_steps = jnp.where(goals_reached, t, batch_steps)
            batch_first_dur = jnp.where(jnp.logical_and(batch_first_flag, batch_rewards), t, batch_first_dur)
            batch_first_flag = jnp.where(batch_rewards, 0, batch_first_flag)
            batch_skill = jnp.where(batch_rewards, batch_first_dur - batch_durations, batch_skill)
            
            if t >= config.steps_per_episode - 1:
                batch_skill = jnp.where(batch_rewards, batch_skill, -config.steps_per_episode)
                batch_final_perf = jnp.where(batch_rewards, config.steps_per_episode - batch_durations, 0)
        
        # Compute fitness
        batch_rewards = jnp.reshape(batch_rewards, (config.population_size, config.landscape_count))
        batch_skill = jnp.divide(batch_skill, config.steps_per_episode)
        batch_skill = jnp.reshape(batch_skill, (config.population_size, config.landscape_count))
        batch_final_perf = jnp.reshape(batch_final_perf, (config.population_size, config.landscape_count))
        
        if config.evolution_mode == 0:
            fitness = vectorized_calculate_multi_objective_fitness(
                batch_rewards, batch_skill, config.mean_reward_weight,
                config.min_reward_weight, config.skill_improvement_weight)
        elif config.evolution_mode == 1:
            fitness = vectorized_calculate_final_performance_fitness(batch_final_perf)
        elif config.evolution_mode == 2:
            if generation <= 100:
                fitness = vectorized_calculate_multi_objective_fitness(
                    batch_rewards, batch_skill, config.mean_reward_weight,
                    config.min_reward_weight, config.skill_improvement_weight)
            else:
                fitness = vectorized_calculate_final_performance_fitness(batch_final_perf)
        
        weights = _centered_rank_transform(fitness)
        w_pos, w_neg = jnp.split(weights, 2, axis=-1)
        weights = w_pos - w_neg
        
        # Update parameters
        grads = jax.tree_util.tree_map(
            lambda pop_p, cen_p: -jnp.mean(weights.reshape([-1] + [1] * (pop_p.ndim - 1)) * 
                                           (pop_p[:config.population_size // 2] - cen_p), axis=0),
            pop_params, center_params)
        
        updates, opt_state = optimizer.update(grads, opt_state, center_params)
        center_params = optax.apply_updates(center_params, updates)
    
    print(f"\nTraining complete: {(time.time() - train_start) / 3600:.2f} hours")
    print(f"Best skill improvement: {best_skill:.4f}")


if __name__ == '__main__':
    main()
