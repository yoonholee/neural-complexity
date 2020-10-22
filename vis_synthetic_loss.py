#%%
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import savefig

# Let's create a function to model and create data
def gauss(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


quad = lambda x: x ** 2
sin = lambda x, scale, bias, freq: scale * np.sin(freq * x - bias)


def true_error(x):
    outputs = [
        -0.1,
        quad(x),
        3 * sin(x, 5e-2, 0, 10),
    ]
    return sum(outputs)


def delta(x):
    outputs = [
        0.05
        - 0.2 * quad(x),
        0.4 * sin(x, 5e-2, 0.2, 20),
        -1.3 * gauss(x, 3e-1, 0.5, 0.2)
    ]
    return sum(outputs)


def delta_2(x):
    outputs = [
        -0.1,
        0.5 * quad(x),
        0.4 * sin(x, 5e-2, 0.2, 20),
        -1 * gauss(x, 3e-1, -0.3, 0.2),
        1.3 * gauss(x, 3e-1, 0.5, 0.2)
    ]
    return sum(outputs)

def delta_3(x):
    outputs = [
        -0.1,
        -0.2 * quad(x),
        sin(x, 5e-2, 0, 10),
        -0.5 * gauss(x, 3e-1, -0.8, 0.2),
        -1 * gauss(x, 3e-1, 0.1, 0.2)
    ]
    return sum(outputs)

y_true = true_error(x)

delta_fn, delta_str = delta_3, "3"
delta_fn, delta_str = delta_2, "2"
delta_fn, delta_str = delta, "1"
x = np.linspace(-1, 1, 1000)
y_delta = delta_fn(x)
y_emp = y_true + y_delta

sns.lineplot(x, y_emp, color="black", label="Empirical Error")
ax = sns.lineplot(x, y_true, color="red", label="True Error")
ax.lines[1].set_linestyle("--")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\mathcal{L}$")
plt.ylim(-0.6, 2.0)
plt.xticks([])
plt.yticks([])
savefig(f"errors_{delta_str}")
plt.show()
plt.clf()

sns.lineplot(x, y_emp, color="black", label="Empirical Error")
ax = sns.lineplot(x, y_true, color="red", label="True Error")
ax.lines[1].set_linestyle("--")
ax.fill_between(x, y_emp, y_true, linewidth=0.0, color="red", alpha=0.2)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\mathcal{L}$")
plt.ylim(-0.6, 2.0)
plt.xticks([])
plt.yticks([])
savefig(f"errors_fill_{delta_str}")
plt.show()
plt.clf()

ax = sns.lineplot(x, 0, color="black")
ax = sns.lineplot(x, y_delta, color="red", label="Gap")
ax.fill_between(x, y_delta, color="red", linewidth=0.0, alpha=0.2)
plt.xticks([])
plt.yticks([0.0])
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\Delta \mathcal{L}$")
plt.ylim(-0.6, 2.0)
savefig(f"gap_{delta_str}")
plt.show()
plt.clf()

# %%


# %%
