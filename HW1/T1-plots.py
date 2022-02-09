import numpy as np
import matplotlib.pyplot as plt

u = np.arange(0, 12.1, 0.1)  # spacing of 0.1

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]


def f(xstar, tau):
    res = 0
    for x, y in data:
        res += np.exp(-(x-xstar)**2/tau) * y
    return res

# plt.plot(u, f(u, 0.01))
# plt.plot(u, f(u, 2))
# plt.plot(u, f(u, 100))
# plt.legend("tau = 0.01", "tau = 2", "tau = 100")
# plt.show()

fig, ax = plt.subplots()
a, = ax.plot(u, f(u, 0.01), label="tau = 0.01")
b, = ax.plot(u, f(u, 2), label = "tau = 2")
c, = ax.plot(u, f(u, 100), label = "tau = 100")
ax.legend(handles = [a,b,c])
ax.set_xlabel('test inputs x*')
ax.set_ylabel('prediction f(x*)')
plt.show()
