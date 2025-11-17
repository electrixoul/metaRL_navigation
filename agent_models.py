from traceback import print_tb
from jax import grad
import jax.numpy as jnp
from jax import jit
import time
import numpy.random as npr
import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, FanOut, Relu, Softplus, Sigmoid, FanInSum
from jax.nn import sigmoid
from functools import partial
from jax import vmap

from flax import linen as nn
from flax.training import train_state
import flax

class MLP(nn.Module):
    in_dims: int
    h1_n: int
  
    @nn.compact
    def __call__(self, x):


        h0 = nn.Dense(self.in_dims)(x)
        h0 = nn.relu(h0)
        # h0 = nn.tanh(h0) + 0.1*h0
        h1 = nn.Dense(self.h1_n)(h0)
        h1 = nn.relu(h1)
        # h1 = nn.tanh(h1) + 0.1*h1
        out1 = nn.Dense(4)(h1)
        out1 = nn.relu(out1)
        # out1 = nn.tanh(out1) + 0.1*out1

        return out1

class MLP1(nn.Module):
    in_dims: int
  
    @nn.compact
    def __call__(self, x):
        h0 = nn.Dense(self.in_dims)(x)
        h0 = nn.relu(h0)
        out1 = nn.Dense(4)(h0)
        out1 = nn.relu(out1)
        return out1

class MLP3(nn.Module):
    in_dims: int
    h1_n: int
  
    @nn.compact
    def __call__(self, x):
        h0 = nn.Dense(self.in_dims)(x)
        h0 = nn.relu(h0)
        h1 = nn.Dense(self.h1_n)(h0)
        h1 = nn.relu(h1)
        h2 = nn.Dense(self.h1_n)(h1)
        h2 = nn.relu(h2)
        out1 = nn.Dense(4)(h2)
        out1 = nn.relu(out1)

        return out1

class MLP_dueling(nn.Module):
    h1_n: int
  
    @nn.compact
    def __call__(self, x):

        h1 = nn.Dense(self.h1_n)(x)
        h1 = nn.relu(h1)
        h20 = nn.Dense(self.h1_n//2)(h1)
        h21 = nn.Dense(self.h1_n//2)(h1)

        h20 = nn.relu(h20)
        h21 = nn.relu(h21)

        out1 = nn.Dense(4)(jnp.concatenate((h20,h21), axis=-1))
        out1 = nn.relu(out1)
        out1 = nn.softmax(out1)

        return out1

class MLP_dueling1(nn.Module):
    h1_n: int
  
    @nn.compact
    def __call__(self, x):

        h1 = nn.Dense(self.h1_n)(x)
        h1 = nn.relu(h1)
        h20 = nn.Dense(self.h1_n//2)(h1)
        h21 = nn.Dense(self.h1_n//2)(h1)

        h20 = nn.relu(h20)
        h21 = nn.relu(h21)

        out1 = nn.Dense(4)(jnp.concatenate((h20,h21), axis=-1))
        out1 = nn.relu(out1)

        return out1

class LSTM(nn.Module):
    in_dims: int = 64
    out_dims: int = 4
    hidden_dims: int = 64

    @nn.compact
    def __call__(self, state, input):
        new_state, output = nn.LSTMCell()(state, input)
        output = nn.Dense(self.out_dims)(output)

        return new_state, output

    def initial_state(self, batch_size):
        # Zero initialization
        return nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size, ), self.hidden_dims)

    @staticmethod
    def state_metrics(state):
        return {}


class GRU(nn.Module):
    in_dims: int = 64
    out_dims: int = 4
    hidden_dims: int = 64

    @nn.compact
    def __call__(self, state, input):
        new_state, output = nn.GRUCell(features=self.hidden_dims)(state, input)
        output = nn.Dense(self.out_dims)(output)

        output = nn.tanh(output)

        return new_state, output

    def initial_state(self, batch_size):
        # Zero initialization
        return jnp.zeros((batch_size, self.hidden_dims))

    def initial_state_rnd(self, batch_size, key):
        # Random initialization
        state = jax.random.normal(key, (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class GRU2(nn.Module):
    in_dims: int = 64
    out_dims: int = 4
    hidden_dims: int = 64

    @nn.compact
    def __call__(self, state, input):
        new_state, output = nn.GRUCell()(state, input)
        output = nn.Dense(self.out_dims)(output)

        output = nn.tanh(output)

        return new_state, output

    def initial_state(self, batch_size):
        # Zero initialization
        return nn.GRUCell.initialize_carry(jax.random.PRNGKey(0), (batch_size, ), self.hidden_dims*2)

    @staticmethod
    def state_metrics(state):
        return {}

class RNN(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64

    @nn.compact
    def __call__(self, state, input):
        
        i1 = jnp.concatenate((state, input), axis=-1)
        new_state = nn.Dense(self.hidden_dims)(i1)
        new_state = nn.relu(new_state)

        out = nn.Dense(self.out_dims)(new_state)
        out = nn.tanh(out)
        
        return new_state, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN_th(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64

    @nn.compact
    def __call__(self, state, input):
        
        i1 = jnp.concatenate((state, input), axis=-1)
        new_state = nn.Dense(self.hidden_dims)(i1)
        new_state = nn.tanh(new_state)

        out = nn.Dense(self.out_dims)(new_state)
        out = nn.tanh(out)
        
        return new_state, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN_th_rs(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64

    def setup(self):
        self.recurrent_layer = nn.Dense(self.hidden_dims)
        self.output_layer = nn.Dense(self.out_dims)
        self.reward_layer = nn.Dense(self.hidden_dims)

    @nn.compact
    def __call__(self, state, input):

        # erase reward
        input0 = jnp.where(input.ndim==1,input.at[4].set(0), input.at[:,4].set(0))
        i10 = jnp.concatenate((state, input), axis=-1)
        i1 = jnp.concatenate((state, input0), axis=-1)

        new_state = self.recurrent_layer(i1)

        reward_signature = self.recurrent_layer(i10) - self.recurrent_layer(i1)
        
        new_state = new_state + nn.sigmoid(new_state - reward_signature)

        new_state = nn.tanh(new_state)

        out = self.output_layer(new_state)
        out = nn.tanh(out)
        
        return new_state, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN_th_rs1(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64

    def setup(self):
        self.recurrent_layer = nn.Dense(self.hidden_dims)
        self.output_layer = nn.Dense(self.out_dims)
        self.reward_layer = nn.Dense(self.hidden_dims)

    @nn.compact
    def __call__(self, state, input):

        # erase reward
        input0 = jnp.where(input.ndim==1,input.at[4].set(0), input.at[:,4].set(0))
        i10 = jnp.concatenate((state, input), axis=-1)
        i1 = jnp.concatenate((state, input0), axis=-1)

        new_state = self.recurrent_layer(i1)
        new_state = nn.tanh(new_state)

        reward_signature = self.recurrent_layer(i10) - self.recurrent_layer(i1)
        
        new_state = new_state + reward_signature

        out = self.output_layer(new_state)
        out = nn.tanh(out)
        
        return new_state, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN_th2(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64

    def setup(self):
        self.hidden_layer = nn.Dense(self.hidden_dims)
        self.hidden_layer1 = nn.Dense(self.hidden_dims)

    @nn.compact
    def __call__(self, state, input):
        
        i1 = jnp.concatenate((state, input), axis=-1)
        new_state = self.hidden_layer(i1)
        new_state = nn.tanh(new_state)

        # inner loop
        new_state = self.hidden_layer1(new_state)
        new_state = nn.tanh(new_state)
        new_state = self.hidden_layer1(new_state)
        new_state = nn.tanh(new_state)
        new_state = self.hidden_layer1(new_state)
        new_state = nn.tanh(new_state)
        new_state = self.hidden_layer1(new_state)
        new_state = nn.tanh(new_state)

        out = nn.Dense(self.out_dims)(new_state)
        out = nn.tanh(out)
        
        return new_state, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN_sg(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64

    @nn.compact
    def __call__(self, state, input):
        
        i1 = jnp.concatenate((state, input), axis=-1)
        new_state = nn.Dense(self.hidden_dims)(i1)
        new_state = nn.sigmoid(new_state*3)
        new_state = 2*(new_state - 0.5)

        out = nn.Dense(self.out_dims)(new_state)
        out = nn.sigmoid(out*3)
        out = 2*(out - 0.5)
        
        return new_state, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN1(nn.Module):
    out_dims: int = 4
    hidden_dims0: int = 100
    hidden_dims1: int = 64
    cross_section: int = 64

    @nn.compact
    def __call__(self, state, input):
        
        i1 = jnp.concatenate((state, input), axis=-1)
        state0 = nn.Dense(self.hidden_dims0)(i1[:,0 : self.hidden_dims0 + self.cross_section])
        state0 = nn.relu(state0)
        state1 = nn.Dense(self.hidden_dims1)(i1[:,self.hidden_dims0 : -1])
        state1 = nn.relu(state1)

        out = nn.Dense(self.out_dims)(state1)
        out = nn.relu(out)

        state = jnp.concatenate((state0, state1), axis=-1)
        
        return state, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims1 + self.hidden_dims0))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN2(nn.Module):
    out_dims: int = 4
    hidden_dims0: int = 100
    hidden_dims1: int = 64
    cross_section: int = 64

    @nn.compact
    def __call__(self, state, input):
        
        i1 = jnp.concatenate((state, input), axis=-1)
        
        state1 = nn.Dense(self.hidden_dims1)(i1[:,self.hidden_dims0 : -1])
        state1 = nn.relu(state1)

        i2 = jnp.concatenate((i1[:,0:self.hidden_dims0], state1[:,0:self.cross_section]), axis=-1)
        state0 = nn.Dense(self.hidden_dims0)(i2)
        state0 = nn.relu(state0)

        out = nn.Dense(self.out_dims)(state1)
        out = nn.relu(out)

        state = jnp.concatenate((state0, state1), axis=-1)
        
        return state, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims1 + self.hidden_dims0))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN3(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64

    def setup(self):

        self.hidden_layer = nn.Dense(self.hidden_dims)
        self.hidden_layer1 = nn.Dense(self.hidden_dims)

    @nn.compact
    def __call__(self, state, input):
        
        i1 = jnp.concatenate((state, input), axis=-1)
        new_state = self.hidden_layer(i1)
        new_state1 = nn.relu(new_state)
        new_state2 = self.hidden_layer1(new_state1)
        new_state2 = nn.relu(new_state2)
        new_state3 = self.hidden_layer1(new_state2)
        new_state3 = nn.relu(new_state3)

        out = nn.Dense(self.out_dims)(new_state3)
        out = nn.relu(out)
        
        return new_state3, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN3_th(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64

    def setup(self):

        self.hidden_layer = nn.Dense(self.hidden_dims)
        self.hidden_layer1 = nn.Dense(self.hidden_dims)

    @nn.compact
    def __call__(self, state, input):
        
        i1 = jnp.concatenate((state, input), axis=-1)
        new_state = self.hidden_layer(i1)
        new_state1 = nn.tanh(new_state)
        new_state2 = self.hidden_layer1(new_state1)
        new_state2 = nn.tanh(new_state2)
        new_state3 = self.hidden_layer1(new_state2)
        new_state3 = nn.tanh(new_state3)

        out = nn.Dense(self.out_dims)(new_state3)
        out = nn.tanh(out)
        
        return new_state3, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN3_lr(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64

    def setup(self):

        self.hidden_layer = nn.Dense(self.hidden_dims)
        self.hidden_layer1 = nn.Dense(self.hidden_dims)

    @nn.compact
    def __call__(self, state, input):
        
        i1 = jnp.concatenate((state, input), axis=-1)
        new_state = self.hidden_layer(i1)
        new_state1 = nn.leaky_relu(new_state)
        new_state2 = self.hidden_layer1(new_state1)
        new_state2 = nn.leaky_relu(new_state2)
        new_state3 = self.hidden_layer1(new_state2)
        new_state3 = nn.leaky_relu(new_state3)

        out = nn.Dense(self.out_dims)(new_state3)
        out = nn.leaky_relu(out)
        
        return new_state3, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN3i(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64
    input_dims: int = 8

    def setup(self):

        print("RNN3i agent")

        self.encoder = nn.Dense(self.input_dims)
        self.hidden_layer = nn.Dense(self.hidden_dims)
        self.hidden_layer1 = nn.Dense(self.hidden_dims)

    @nn.compact
    def __call__(self, state, input):
        
        i0 = self.encoder(input)
        i0 = nn.tanh(i0)
        i1 = jnp.concatenate((state, i0), axis=-1)
        new_state = self.hidden_layer(i1)
        new_state1 = nn.relu(new_state)
        new_state2 = self.hidden_layer1(new_state1)
        new_state2 = nn.relu(new_state2)
        new_state3 = self.hidden_layer1(new_state2)
        new_state3 = nn.relu(new_state3)

        out = nn.Dense(self.out_dims)(new_state3)
        out = nn.relu(out)
        
        return new_state3, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN3i1(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64
    input_dims: int = 8

    def setup(self):

        self.encoder = nn.Dense(32)
        self.encoder1 = nn.Dense(self.input_dims)
        self.hidden_layer = nn.Dense(self.hidden_dims)
        self.hidden_layer1 = nn.Dense(self.hidden_dims)

    @nn.compact
    def __call__(self, state, input):
        
        i0 = self.encoder(input)
        i0 = nn.tanh(i0)
        i1 = self.encoder1(i0)
        i1 = nn.tanh(i1)
        i2 = jnp.concatenate((state, i1), axis=-1)
        new_state = self.hidden_layer(i2)
        new_state1 = nn.relu(new_state)
        new_state2 = self.hidden_layer1(new_state1)
        new_state2 = nn.relu(new_state2)
        new_state3 = self.hidden_layer1(new_state2)
        new_state3 = nn.relu(new_state3)

        out = nn.Dense(self.out_dims)(new_state3)
        out = nn.relu(out)
        
        return new_state3, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}

class RNN3_sg(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64

    def setup(self):

        self.hidden_layer = nn.Dense(self.hidden_dims)
        self.hidden_layer1 = nn.Dense(self.hidden_dims)

    @nn.compact
    def __call__(self, state, input):
        
        i1 = jnp.concatenate((state, input), axis=-1)
        new_state = self.hidden_layer(i1)
        new_state1 = nn.sigmoid(new_state)
        new_state1 = new_state1 - 0.5
        new_state2 = self.hidden_layer1(new_state1)
        new_state2 = nn.sigmoid(new_state2)
        new_state2 = new_state2 - 0.5
        new_state3 = self.hidden_layer1(new_state2)
        new_state3 = nn.sigmoid(new_state3)
        new_state3 = new_state3 - 0.5

        out = nn.Dense(self.out_dims)(new_state3)
        out = nn.sigmoid(out)
        out = out - 0.5
        
        return new_state3, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}


class RNN_th3(nn.Module):
    out_dims: int = 4
    hidden_dims: int = 64

    @nn.compact
    def __call__(self, state, input):
        
        i1 = jnp.concatenate((state, input), axis=-1)
        new_state = nn.Dense(self.hidden_dims)(i1)
        new_state = nn.tanh(new_state)

        out = nn.Dense(self.out_dims)(new_state)
        out = nn.tanh(out)
        
        return new_state, out

    def initial_state(self, batch_size):
        # Zero initialization
        state = jnp.zeros((batch_size, self.hidden_dims*2))
        return state

    def initial_state_rnd(self, batch_size, key):
        # Zero initialization
        state = 1.5 * jax.random.normal(jax.random.PRNGKey(key), (batch_size, self.hidden_dims))
        return state

    @staticmethod
    def state_metrics(state):
        return {}
