import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def cost(x):
    return x**2 + 5 * np.sin(x)


def grad(x):
    return 2 * x + 5 * np.cos(x)


def GD(x_init, eta):
    x = [x_init]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return x, it


x_gd, it = GD(-10, 0.1)


X = np.linspace(-10, 10, 1000)
y = cost(X)

fig, ax = plt.subplots()


def animation(i):
    ax.clear()
    ax.plot(X, y)
    ax.plot(x_gd[i], cost(x_gd[i]), "ro")
    if i > 0:
        ax.plot(x_gd[i - 1], cost(x_gd[i - 1]), "bo")
        ax.plot([x_gd[i - 1], x_gd[i]], [cost(x_gd[i - 1]), cost(x_gd[i])], "r")
    ax.set_xlabel(
        "iter:{}  cost:{:.4f}  grad:{:.4f}".format(i, cost(x_gd[i]), grad(x_gd[i]))
    )


ani = FuncAnimation(fig, animation, frames=it, interval=500, repeat=False)
plt.show()
