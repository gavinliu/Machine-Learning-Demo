import numpy as np

data = np.loadtxt('data.txt', delimiter=',')

X = np.c_[np.ones(data.shape[0]), data[:, 0]]
y = np.c_[data[:, 1]]


def computeCost(X, y, theta=[[0], [0]]):
    m = y.size
    h = X.dot(theta)

    return 1.0 / (2 * m) * (np.sum(np.square(h - y)))


def gradientDescent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=100):
    m = y.size
    J_history = np.zeros(num_iters)

    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1.0 / m) * (X.T.dot(h - y))

        J_history[iter] = computeCost(X, y, theta)
    return (theta, J_history)


theta, Cost_J = gradientDescent(X, y)

print(theta)
