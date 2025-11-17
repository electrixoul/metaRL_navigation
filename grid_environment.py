import copy
from traceback import print_tb
from jax import grad
import jax.numpy as jnp
from jax import jit
import time
import numpy as np
import numpy.random as npr
import jax
import jax.numpy as jnp
from jax import device_put
from jax import jit, grad, lax, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, FanOut, Relu, Softplus, Sigmoid, FanInSum
from jax.nn import sigmoid
from functools import partial
from jax import vmap
from flax import linen as nn
from flax.training import train_state
from flax import struct
from jax import lax

from jax import tree_util
from jax.tree_util import tree_structure
from jax.tree_util import tree_flatten, tree_unflatten


@jax.jit
def goal_determinant(start, goal):
    return jnp.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)<1
batch_compute_goal_reached = vmap(goal_determinant, in_axes=(0, 0))

'''
    [runtime] step physics for a single environment
'''
@partial(jax.jit)
def step_physics_one_grid(grid_data, start, goal, action, action_list):

    s_next = start + action_list[action]

    width, height = grid_data.shape[0], grid_data.shape[1]

    # check if new position is an obstacle
    # check if new position is out of boundry
    obs_next = (grid_data[s_next[0], s_next[1]] == 0)
    x_inbound = (s_next[0] < 0) | (s_next[0] >= width) | obs_next
    y_inbound = (s_next[1] < 0) | (s_next[1] >= height) | obs_next

    s_next = s_next.at[0].set(x_inbound * start[0] + (1-x_inbound) * s_next[0])
    s_next = s_next.at[1].set(y_inbound * start[1] + (1-y_inbound) * s_next[1])

    # check if goal is reached
    goal_reached = goal_determinant(s_next, goal)

    return s_next, goal_reached

'''
    [runtime] vmap function for step_physics_one_grid
'''
batch_step_physics = vmap(step_physics_one_grid, in_axes=(0, 0, 0, 0, None))

@partial(jax.jit, static_argnums=(2,))
def make_ranged_id(key_, id_range = 20, quant = 100):
    return jax.random.randint(key_, shape=(quant,), minval=0, maxval=id_range)

make_ranged_id_vmap = vmap(make_ranged_id, in_axes=(0, 0, None))

@partial(jax.jit)
def reset_state(batched_goal_reached, states, initial_states):

    tile_batched_goal_reached = jnp.tile(batched_goal_reached, (2,1)).T
    states = jnp.where(tile_batched_goal_reached==True, initial_states, states)

    return states

@partial(jax.jit)
def reset_target(key_, ranged_ids, batched_goal_reached, rnd_goal_collection, batched_goals):

    ''' generate random indices '''
    new_key, subkey = random.split(key_)
    rand_id = jax.random.randint(new_key, shape=(1,), minval=0, maxval=100)
    ranged_ids_choosen = jnp.squeeze(ranged_ids[:,rand_id], axis=1)
    ranged_ids_choosen = jnp.expand_dims(ranged_ids_choosen, axis=1)
    rnd_goals_choosen = jnp.take_along_axis(rnd_goal_collection, ranged_ids_choosen[:,:,None], axis=1)
    rnd_goals_choosen = jnp.squeeze(rnd_goals_choosen, axis=1)

    ''' apply mask to rnd_goals_choosen '''
    tile_batched_goal_reached = jnp.tile(batched_goal_reached, (2,1)).T
    batched_goals = jnp.where(tile_batched_goal_reached==True, rnd_goals_choosen, batched_goals)

    return subkey, batched_goals, rnd_goals_choosen

@partial(jax.jit, static_argnums=(2,3,))
def get_rnd_goal_collection(rng_key, env, width, height, num_empty_space):

    def get_rnd_goal_and_mark(rng_key, env, width, height):

        env_linear = jnp.reshape(jnp.transpose(env), (width * height))
        free_space_index = jnp.flatnonzero(env_linear, size = (width * height), fill_value = -1)
        free_space_index_0 = jnp.where(free_space_index == -1, 0, free_space_index)
        num_free_space = jnp.count_nonzero(free_space_index_0)
        state_index = jax.random.randint(rng_key, (1,), 0, num_free_space)
        # get end from state_index
        goal_x = free_space_index[state_index[0]] % width
        goal_y = free_space_index[state_index[0]] // width
        
        return jnp.array([goal_x, goal_y], dtype=jnp.int8)
    
    collection_size = num_empty_space

    xs = jnp.array([i for i in range(width*height)], dtype=jnp.int16)
    zero_goal = jnp.array([-1, -1], dtype=jnp.int8)

    def scan_f(env, x):
        goal_new = get_rnd_goal_and_mark(rng_key, env, width, height)
        env = jnp.where(x < collection_size, env.at[goal_new[0], goal_new[1]].set(0), env)
        goal = jnp.where(x < collection_size, goal_new, zero_goal)
        return env, goal

    env, goal_collection = jax.lax.scan(scan_f, env, xs, length = width * height)

    return goal_collection
    
get_rnd_goal_collection_vmap = vmap(get_rnd_goal_collection, in_axes=(0, 0, None, None, 0))

@partial(jax.jit, static_argnums=(1,2,))
def count_free_space(env, width, height):
    env_linear = jnp.reshape(jnp.transpose(env), (width * height))
    free_space_index = jnp.flatnonzero(env_linear, size = (width * height), fill_value = -1)
    free_space_index_0 = jnp.where(free_space_index == -1, 0, free_space_index)
    num_free_space = jnp.count_nonzero(free_space_index_0)
    return num_free_space

count_free_space_vmap = vmap(count_free_space, in_axes=(0, None, None))

@partial(jax.jit, static_argnums=(2,3,))
def get_rnd_state(rng_key, env, width, height):

    env_linear = jnp.reshape(jnp.transpose(env), (width * height))
    free_space_index = jnp.flatnonzero(env_linear, size = (width * height), fill_value = -1)
    free_space_index_0 = jnp.where(free_space_index == -1, 0, free_space_index)
    num_free_space = jnp.count_nonzero(free_space_index_0)

    state_index = jax.random.randint(rng_key, (2,), 0, num_free_space)

    # get start and end from state_index
    start_x = free_space_index[state_index[0]] % width
    start_y = free_space_index[state_index[0]] // width

    env_linear = env_linear.at[free_space_index[state_index[0]]].set(0)
    free_space_index = jnp.flatnonzero(env_linear, size = (width * height), fill_value = -1)
    free_space_index_0 = jnp.where(free_space_index == -1, 0, free_space_index)
    num_free_space = jnp.count_nonzero(free_space_index_0)

    state_index = jax.random.randint(rng_key, (2,), 0, num_free_space)

    end_x = free_space_index[state_index[1]] % width
    end_y = free_space_index[state_index[1]] // width

    start = jnp.array([start_x, start_y])
    end = jnp.array([end_x, end_y])

    return start, end
get_rnd_state_vmap = vmap(get_rnd_state, in_axes=(0, 0, None, None))

def count_landscapes(landscapes):
    return jnp.shape(landscapes)[0]

""" ideal observation
"""
@partial(jax.jit)
def get_ideal_obs(grid_data, start, goal, last_reward):
    start_x = start[0]
    start_y = start[1]
    goal_x = goal[0]
    goal_y = goal[1]
    new_grid_data = grid_data.at[goal_x, goal_y].set(-2)
    # new_grid_data = grid_data
    local_obs = jax.lax.dynamic_slice(new_grid_data, (start_x-1,start_y-1), (3,3))
    local_obs_flat = jnp.reshape(local_obs, (9,))

    # replace 0 with 1 and 1 with 0
    local_obs_flat = jnp.where(local_obs_flat == 0, -1, local_obs_flat)
    local_obs_flat = jnp.where(local_obs_flat == 1, 0, local_obs_flat)
    local_obs_flat = jnp.where(local_obs_flat == -1, 1, local_obs_flat)

    # add last reward
    local_obs_flat = jnp.append(local_obs_flat, last_reward)
    # local_obs_flat = jnp.append(local_obs_flat, 0)
    
    return local_obs_flat

get_ideal_obs_vmap = vmap(get_ideal_obs, in_axes=(0, 0, 0, 0))


def get_ideal_obs_rf(grid_data, start, goal, last_reward):
    """Returns the ideal observation for the agent

    The ideal observation is the observation that the agent would get if it had
    perfect information about the environment. In this case, the agent knows the
    exact location of the goal, so all it needs to know is what is in the 3x3
    grid around its current position. This function returns the ideal observation
    for the agent, given its current position and the location of the goal.

    Args:
        grid_data: a 2D array representing the current state of the environment
        start: a tuple representing the current position of the agent
        goal: a tuple representing the location of the goal
        last_reward: the reward the agent received in the previous time step

    Returns:
        an array of length 10 representing the ideal observation for the agent
    """
    start_x = start[0]
    start_y = start[1]
    goal_x = goal[0]
    goal_y = goal[1]
    new_grid_data = grid_data
    local_obs = jax.lax.dynamic_slice(new_grid_data, (start_x-1,start_y-1), (3,3))
    local_obs_flat = jnp.reshape(local_obs, (9,))

    # replace 0 with 1 and 1 with 0
    local_obs_flat = jnp.where(local_obs_flat == 0, -1, local_obs_flat)
    local_obs_flat = jnp.where(local_obs_flat == 1, 0, local_obs_flat)
    local_obs_flat = jnp.where(local_obs_flat == -1, 1, local_obs_flat)

    # add last reward
    local_obs_flat = jnp.append(local_obs_flat, 0)
    
    return local_obs_flat
get_ideal_obs_vmap_rf = vmap(get_ideal_obs_rf, in_axes=(0, 0, 0, 0))

'''
    [data-preparation] create grid environment, agent state, food state
'''
@partial(jax.jit, static_argnums=(1,2,))
def create_env_and_state(env_id, width, height, landscape_jnp):

    env = jnp.zeros((width, height), dtype=jnp.int8)
    for i in range(width):
        for j in range(height):
            env = env.at[i, j].set(landscape_jnp[env_id, i, j])
    return env
'''
    [data-preparation] batched version of create_env_and_state
'''
batch_create_env = vmap(create_env_and_state, in_axes=(0, None, None, None))

class GridEnv:
    def __init__(self, 
        width: int = 4,
        height: int = 9,
        landscapes: list = [],
        headless: bool = False, 
        num_envs_per_landscape: int = 1,
        empty: bool = False,
        global_obs_mode: bool = False,
        num_lidar_bins: int = 128,
        reward_free: bool = False,
        ):

        self.reward_free = reward_free

        self.width = width
        self.height = height
        self.headless = headless
        self.num_lidar_bin = num_lidar_bins
        self.empty = empty
        self.scale_ = 10
        self.max_steps = 300
        self.global_obs_mode = global_obs_mode
        self.min_free_space = 20

        self.act = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]])

        self.num_landscapes = count_landscapes(landscapes)
        self.num_envs_per_landscape = num_envs_per_landscape
        self.num_envs = self.num_envs_per_landscape * self.num_landscapes

        # set landscape
        if jnp.shape(landscapes)[-1] == self.width * self.height:
            self.landscapes = copy.deepcopy(landscapes)
        else:
            return

        self.landscape_jnp = jnp.zeros((jnp.shape(landscapes)[0], self.width, self.height), dtype=jnp.int8)
        for n in range(jnp.shape(landscapes)[0]):
            for i in range(self.width):
                for j in range(self.height):
                    self.landscape_jnp = self.landscape_jnp.at[n, i, j].set(self.landscapes[n][j * self.width + i])
        
        # print(self.landscape_jnp)

        vec_env_ids = jnp.array([[i for i in range(self.num_landscapes)]], dtype=jnp.int32)
        self.vec_env_ids = jnp.repeat(vec_env_ids, self.num_envs_per_landscape, axis=0)
        self.vec_env_ids = jnp.reshape(self.vec_env_ids, (self.num_envs,))

        print("generating rng keys")
        self.key_ = jax.random.PRNGKey(npr.randint(0, 1000000))
        self.env_keys = jax.random.split(self.key_, self.num_envs)
        self.env_keys_landscape = jax.random.split(self.key_, self.num_landscapes)

        print("generating environments")
        self.batched_envs = batch_create_env(self.vec_env_ids, self.width, self.height, self.landscape_jnp)
        self.batched_states, self.batched_goals = get_rnd_state_vmap(self.env_keys, self.batched_envs, self.width, self.height)
        self.batched_goal_reached = batch_compute_goal_reached(self.batched_states, self.batched_goals)

        # backup self.batched_states, self.batched_goals as inital states
        self.init_batched_states, self.init_batched_goals = jnp.copy(self.batched_states), jnp.copy(self.batched_goals)
        self.last_batched_goal_reached = jnp.copy(self.batched_goal_reached)

        '''
            compute obs for batched_envs
        '''
        if not self.reward_free:
            self.concat_obs = get_ideal_obs_vmap(self.batched_envs, self.batched_states, self.batched_goals, self.last_batched_goal_reached)
        else:
            self.concat_obs = get_ideal_obs_vmap_rf(self.batched_envs, self.batched_states, self.batched_goals, self.last_batched_goal_reached)

        self.num_free_spaces = count_free_space_vmap(self.landscape_jnp, self.width, self.height)
        self.num_free_spaces = jnp.expand_dims(self.num_free_spaces, axis=0)
        self.num_free_spaces = jnp.repeat(self.num_free_spaces, self.num_envs_per_landscape, axis=0)
        self.num_free_spaces = jnp.reshape(self.num_free_spaces, (self.num_envs,))
        print("num_free_spaces", self.num_free_spaces)

        start = time.time()
        ''' get rnd goals for all envs '''
        self.rnd_goal_collection = get_rnd_goal_collection_vmap(self.env_keys, self.batched_envs, self.width, self.height, self.num_free_spaces)
        print("time taken for rnd goal collection", time.time() - start)
        print(self.rnd_goal_collection.shape)
        self.ranged_ids = make_ranged_id_vmap(self.env_keys, self.num_free_spaces, self.width*self.height)
        print("shape of self.ranged_ids", self.ranged_ids.shape)

    '''
        [runtime] set new landscapes
    '''
    def set_landscapes(self, landscapes):

        self.landscapes = copy.deepcopy(landscapes)
        for n in range(jnp.shape(landscapes)[0]):
            for i in range(self.width):
                for j in range(self.height):
                    self.landscape_jnp = self.landscape_jnp.at[n, i, j].set(self.landscapes[n][j * self.width + i])

        self.batched_envs = batch_create_env(self.vec_env_ids, self.width, self.height, self.landscape_jnp)

        self.batched_states, self.batched_goals = get_rnd_state_vmap(self.env_keys, self.batched_envs, self.width, self.height)
        self.init_batched_states, self.init_batched_goals = jnp.copy(self.batched_states), jnp.copy(self.batched_goals)
        self.batched_goal_reached = batch_compute_goal_reached(self.batched_states, self.batched_goals)
        self.last_batched_goal_reached = jnp.copy(self.batched_goal_reached)

        self.reset()

    '''
        [runtime] step function
        batched_actions : actions to be applied to all environments
    '''
    def step(self, batched_actions, reset_during_step = True):

        ''' reset state '''
        self.batched_states = reset_state(self.batched_goal_reached, self.batched_states, self.init_batched_states)

        # print(self.init_batched_states)
        
        self.batched_states, self.batched_goal_reached = batch_step_physics(self.batched_envs, self.batched_states, self.batched_goals, batched_actions, self.act)

        # get instant observation
        if not self.reward_free:
            self.concat_obs = get_ideal_obs_vmap(self.batched_envs, self.batched_states, self.batched_goals, self.last_batched_goal_reached)
        else:
            self.concat_obs = get_ideal_obs_vmap_rf(self.batched_envs, self.batched_states, self.batched_goals, self.last_batched_goal_reached)

        self.last_batched_goal_reached = jnp.copy(self.batched_goal_reached)

        # print(self.concat_obs)

        return self.batched_goal_reached, self.concat_obs
    
    '''
        [runtime] set target for env[0]
    '''
    def set_target(self, target_x, target_y):
        self.batched_goals = jnp.array([[target_x, target_y] for i in range(self.num_envs)])
        self.batched_goal_reached = batch_compute_goal_reached(self.batched_states, self.batched_goals)

    '''
        [runtime] render
    '''
    def render(self, env_id = 0, food_x_ = -1, food_y_ = -1):
        
        if self.headless == True:
            return

        grid = self.batched_envs[env_id]

        state_x = int(self.batched_states[env_id][0])
        state_y = int(self.batched_states[env_id][1])

        food_x = int(self.batched_goals[env_id][0])
        food_y = int(self.batched_goals[env_id][1])

        if (food_x_ != -1 and food_y_ != -1):
            food_x = food_x_
            food_y = food_y_

        goal_reached = self.batched_goal_reached[env_id]

        grid_size_display = 20
        width, height = grid.shape[0], grid.shape[1]
        img = np.zeros((width * grid_size_display, height * grid_size_display, 3), np.uint8)
        
        for j in range(width):
            for i in range(height):
                if grid[j,i] == 1:
                    cv2.rectangle(img, (i * grid_size_display, j * grid_size_display), (i * grid_size_display + grid_size_display, j * grid_size_display + grid_size_display), (255, 255, 255), -1)
                    # draw border with color(100,100,100)
                    cv2.rectangle(img, (i * grid_size_display, j * grid_size_display), (i * grid_size_display + grid_size_display, j * grid_size_display + grid_size_display), (100, 100, 100), 1)
                else:
                    cv2.rectangle(img, (i * grid_size_display, j * grid_size_display), (i * grid_size_display + grid_size_display, j * grid_size_display + grid_size_display), (0, 0, 0), -1)
                    # draw border with color(100,100,100)
                    cv2.rectangle(img, (i * grid_size_display, j * grid_size_display), (i * grid_size_display + grid_size_display, j * grid_size_display + grid_size_display), (100, 100, 100), 1)
                if j == state_x and i == state_y:
                    cv2.circle(img, (i * grid_size_display + int(grid_size_display/2), j * grid_size_display + int(grid_size_display/2)), 7, (0, 0, 255), -1, cv2.LINE_AA)
        
        img_concat_obs = np.zeros((3 * grid_size_display, 3 * grid_size_display, 3), np.uint8)
        # draw concat_obs using same method
        for j in range(3):
            for i in range(3):
                if self.concat_obs[env_id][j*3 + i] == 1:
                    cv2.rectangle(img_concat_obs, (i * grid_size_display, j * grid_size_display), (i * grid_size_display + grid_size_display, j * grid_size_display + grid_size_display), (255, 255, 255), -1)
                    # draw border with color(100,100,100)
                    cv2.rectangle(img_concat_obs, (i * grid_size_display, j * grid_size_display), (i * grid_size_display + grid_size_display, j * grid_size_display + grid_size_display), (100, 100, 100), 1)
                elif self.concat_obs[env_id][j*3 + i] == 0:
                    cv2.rectangle(img_concat_obs, (i * grid_size_display, j * grid_size_display), (i * grid_size_display + grid_size_display, j * grid_size_display + grid_size_display), (0, 0, 0), -1)
                    # draw border with color(100,100,100)
                    cv2.rectangle(img_concat_obs, (i * grid_size_display, j * grid_size_display), (i * grid_size_display + grid_size_display, j * grid_size_display + grid_size_display), (100, 100, 100), 1)
                elif self.concat_obs[env_id][j*3 + i] == 3:
                    cv2.rectangle(img_concat_obs, (i * grid_size_display, j * grid_size_display), (i * grid_size_display + grid_size_display, j * grid_size_display + grid_size_display), (0, 100, 0), -1)
                    # draw border with color(100,100,100)
                    cv2.rectangle(img_concat_obs, (i * grid_size_display, j * grid_size_display), (i * grid_size_display + grid_size_display, j * grid_size_display + grid_size_display), (100, 100, 100), 1)

        # put with a dot on food position
        cv2.circle(img, (food_y * grid_size_display + grid_size_display//2, food_x * grid_size_display + grid_size_display//2), 7, (0,100,0), -1, cv2.LINE_AA)

        # put with a dot on food position
        cv2.circle(img, (food_y * grid_size_display + grid_size_display//2, food_x * grid_size_display + grid_size_display//2), 7, (0,100,0), -1, cv2.LINE_AA)

        # if goal_reached:
        #     cv2.putText(img, "Goal Reached", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2, cv2.LINE_AA)

        # # create a large image and put img and img_concat_obs together
        # img_large = np.zeros((width * grid_size_display, height * grid_size_display + 3 * grid_size_display + 50, 3), np.uint8)
        # img_large[:] = (100, 100, 100)
        # img_large[0:width * grid_size_display, 0:height * grid_size_display] = img
        # img_large[grid_size_display*4:3 * grid_size_display+grid_size_display*4, height * grid_size_display + 25:height * grid_size_display + 3 * grid_size_display + 25] = img_concat_obs
        
        # draw a size=3*grid_size_display box around the current position
        # cv2.rectangle(img_large, ((state_y-1) * grid_size_display, (state_x-1) * grid_size_display), ((state_y-1) * grid_size_display + 3*grid_size_display, (state_x-1) * grid_size_display + 3*grid_size_display), (0, 200, 0), 3)

        # return img_large
        return img

    '''
        [runtime] reset
    '''
    # TODO : Batched-Reset
    def reset(self):

        new_key, subkey = jax.random.split(self.key_)
        self.key_ = new_key
        self.env_keys = jax.random.split(subkey, self.num_envs)
        self.env_keys_landscape = jax.random.split(self.key_, self.num_landscapes)

        self.batched_states, self.batched_goals = get_rnd_state_vmap(self.env_keys, self.batched_envs, self.width, self.height)
        self.init_batched_states, self.init_batched_goals = jnp.copy(self.batched_states), jnp.copy(self.batched_goals)

        self.batched_goal_reached = batch_compute_goal_reached(self.batched_states, self.batched_goals)
        self.last_batched_goal_reached = jnp.copy(self.batched_goal_reached)
        # get instant observation
        self.concat_obs = get_ideal_obs_vmap(self.batched_envs, self.batched_states, self.batched_goals, self.last_batched_goal_reached)

    '''
        flat and unflat operation for pytree
    '''
    def _tree_flatten(self):
        # dynamic values
        children = ()
        # static values
        aux_data = {
                    'width': self.width,
                    'height': self.height,
                    'landscape': self.landscapes,
                    'headless': self.headless,
                    'num_envs': self.num_envs,
                    'empty': self.empty,
                    'global_obs_mode': self.global_obs_mode,
                    'num_lidar_bins': self.num_lidar_bin,
                    'reward_free': self.reward_free,
                    }
        return (children, aux_data)
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(GridEnv,
                               GridEnv._tree_flatten,
                               GridEnv._tree_unflatten)
