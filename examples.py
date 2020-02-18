import numpy as np
from scipy.stats import ortho_group


class Example1:  # Quadratic function
    def __init__(self, seed, n=500):
        self.n = n
        self.seed = seed
        if self.n == 500:
            self.A_samples = np.load("params/params_ex1/A_samples_n500.npy")
            self.A = self.A_samples[self.seed]
        else:
            np.random.seed(self.seed)
            self.Q = ortho_group.rvs(dim = self.n)
            self.S = np.diag(np.random.uniform(10e-3, 1, self.n))
            self.A = np.dot(np.dot(self.Q, self.S), self.Q.T)
        self.p0 = np.zeros(self.n)
        self.x0 = np.ones(self.n)
        self.s0 = 0
        self.t0 = 0

    def f(self, x):
        return 0.5 * np.dot(np.dot(x, self.A), x)

    def gradf(self, x):
        return np.dot(x, self.A)

    def f2D(self, x1, x2):
        return 0.5*(self.A[0][0] * x1 ** 2 + self.A[1][0] * x1 * x2 + self.A[0][1] * x1 * x2 + self.A[1][1] * x2 ** 2)


class Example2:  # Correlated quadratic function
    def __init__(self, seed, n=50):
        self.n = n
        self.seed = seed
        np.random.seed(self.seed)

        self.B = [[np.sqrt((i + 1) * (j + 1)) / 2 ** (abs(i - j)) for i in range(self.n)] for j in range(self.n)]
        self.p0 = np.zeros(self.n)
        self.x0 = np.random.uniform(-1, 1, self.n)
        self.s0 = 0
        self.t0 = 0

    def f(self, x):
        return 0.5 * np.dot(np.dot(x, self.B), x)

    def gradf(self, x):
        return np.dot(x, self.B)

    def f2D(self, x1, x2):
        return 0.5*(self.B[0][0] * x1 ** 2 + self.B[1][0] * x1 * x2 + self.B[0][1] * x1 * x2 + self.B[1][1] * x2 ** 2)


class Example3:  # Camelback function
    def __init__(self, x0, seed):
        self.n = 2
        self.seed = seed
        np.random.seed(self.seed)
        self.p0 = np.array([0, 0])
        self.x0 = x0
        self.s0 = 0
        self.t0 = 0

    def f(self, x):
        return 2*x[0]**2 - 1.05*x[0]**4 + 1/6*x[0]**6 + x[0]*x[1] + x[1]**2

    def gradf(self, x):
        return np.array([4*x[0] - 4.2*x[0]**3 + x[0]**5 + x[1], x[0] + 2*x[1]])

    def f2D(self, x1, x2):
        return 2*x1**2 - 1.05*x1**4 + 1/6*x1**6 + x1*x2 + x2**2


class Example4:  # Rosenbrock function
    def __init__(self, x0, seed, n=100):
        self.n = n
        self.seed = seed
        np.random.seed(self.seed)
        self.p0 = np.zeros(self.n)
        self.x0 = x0
        self.s0 = 0
        self.t0 = 0

    def f(self, x):
        suma = 0
        for i in range(self.n-1):
            suma += 100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2
        return suma

    def gradf(self, x):
        gradV = np.empty(self.n, dtype=np.float64)
        gradV[0] = -400*x[0]*(x[1] - x[0]**2) - 2*(1-x[0])
        gradV[-1] = 200*(x[self.n-1] - x[self.n-2]**2)

        for i in range(self.n-2):
            gradV[i+1] = 200*(x[i+1] - x[i]**2) - 400*(x[i+2] - x[i+1]**2)*x[i+1] - 2*(1-x[i+1])
        return gradV

    def f2D(self, x1, x2):
        return 100*(x2 - x1**2)**2 + (1-x1)**2 + 100*(1 - x2**2)**2 + (1-x2)**2
