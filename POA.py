import numpy as np
from scipy.signal import hilbert

def envelope_entropy(signal, K, alpha):
    """计算信号的包络熵"""
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_normalized = amplitude_envelope / np.sum(amplitude_envelope)
    return -np.sum(amplitude_normalized ** alpha * np.log(amplitude_normalized ** alpha))

def fitness_function(candidate_solution, K, alpha):
    """适应度函数，使用包络熵作为评价指标"""
    return envelope_entropy(candidate_solution, K, alpha)

# Example:
signal = np.random.rand(100)

initial_K = 10
initial_alpha = 0.5

K = initial_K
alpha = initial_alpha


fitness_value = fitness_function(signal, K, alpha)
print("Fitness value (initial):", fitness_value)


K = 20
alpha = 0.7

fitness_value = fitness_function(signal, K, alpha)
print("Fitness value (updated):", fitness_value)
