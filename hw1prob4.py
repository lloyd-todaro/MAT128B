import numpy as np
import math

def bisection_method(f:np.array, a:float, b:float, epsilon:float)-> float:
    '''
    f is a numpy array of samples of a continuous function, indexed by
    round(x) (i.e. f[round(x)] approximates f(x))
    we want to find a root in [a,b]
    we continue running until |x_k-x_{k-1}|<epsilon
    outputting an x value for a root of f in [a,b], along with the
    sequence of midpoints visited at each iteration
    '''
    if np.sign(f[math.floor(a)]) == np.sign(f[math.floor(b)]):
        raise ValueError("Signs must be opposite at ends of the interval to use bisection method")

    x_history = []
    x_prev = a
    while True:
        midpoint = (a + b) / 2
        x_history.append(midpoint)

        if f[round(midpoint)] == 0 or abs(midpoint - x_prev) < epsilon:
            return midpoint, x_history

        if np.sign(f[round(midpoint)]) == np.sign(f[math.floor(a)]):
            a = midpoint
        else:
            b = midpoint
        x_prev = midpoint

def newton_method(f:np.array, f_prime:np.array, x0:float, epsilon:float, tolerance: int)-> float:
    '''
    f and f_prime are numpy arrays of samples of a differentiable function
    and its derivative, indexed by round(x) (i.e. f[round(x)] approximates f(x))
    x0 is the initial guess
    we continue running until |x_k-x_{k-1}|<epsilon
    OR until we have run more than tolerance iterations
    outputting an x value for a root of f, along with the sequence of
    iterates x_k visited
    '''
    x_history = [x0]
    x_k = x0

    for _ in range(tolerance):
        f_val = f[round(x_k)]
        f_prime_val = f_prime[round(x_k)]
        if f_prime_val == 0:
            raise ZeroDivisionError("f_prime is zero at this iterate; Newton's method fails")

        x_next = x_k - f_val / f_prime_val
        x_history.append(x_next)

        if abs(x_next - x_k) < epsilon:
            return x_next, x_history

        x_k = x_next

    return x_k, x_history

    