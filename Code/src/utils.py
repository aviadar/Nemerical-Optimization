import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def report(iteration, x, fx, x_diff, fx_diff):
    print(f'{iteration=}, {x=}, {fx=}, {x_diff=}, {fx_diff=}')


def final_report(success, last_x):
    print(f'the convergence is {success} at last x={last_x}')


def plot_val_hist(val_hist, title):
    plt.plot(val_hist)
    plt.xlabel('Iterations')
    plt.ylabel('objective function value')
    plt.title(title)
    plt.show()


def plot_contours_paths(f, x_hist):
    x1 = np.linspace(-3, 3, 1000)
    x2 = np.linspace(-8, 3, 1000)
    X, Y = np.meshgrid(x1, x2)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f.evaluate(np.array([[X[i, j]], [Y[i, j]]]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50)

    x1_hist = []
    x2_hist = []
    z_hist = []
    for x in x_hist.T:
        fx = f.evaluate(x)
        x1_hist.append(x[0])
        x2_hist.append(x[1])
        z_hist.append(fx)

    ax.plot(x1_hist, x2_hist, z_hist, 'o', markersize=12, label='path')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.legend()
    colors = cm.Pastel1(range(x_hist.shape[1]))
    fig, ax1 = plt.subplots()
    CS = ax1.contour(X, Y, Z, 30)
    ax1.clabel(CS, inline=True, fontsize=10)
    for iteration, x in enumerate(x_hist.T):
        ax1.plot(x[0], x[1], 'o', color=colors[iteration], label='path'+str(iteration))

    plt.xlabel('x1')
    plt.ylabel('x2')
    ax1.legend()

    plt.show()


def plot_qp(f, x_hist):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x0 = np.linspace(0, 1, 100)
    x1 = np.linspace(0, 1, 100)

    X, Y = np.meshgrid(x0, x1)
    # Z = np.maximum(0, 1 - X - Y)
    Z = 1-X-Y
    C = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            C[i, j] = f.evaluate(np.array([[X[i, j]], [Y[i, j]], [Z[i, j]]]))

    valid_indicies = np.ones(X.shape)
    valid_indicies[X < 0] = np.nan
    valid_indicies[Y < 0] = np.nan
    valid_indicies[Z < 0] = np.nan
    X[np.isnan(valid_indicies)] = np.nan
    Y[np.isnan(valid_indicies)] = np.nan
    Z[np.isnan(valid_indicies)] = np.nan
    # ax.contour3D(X,Y,Z,cmap=valid_indicies )
    ax.plot_surface(X, Y, Z, alpha=0.5)
    # plt.pcolor(X,Y,Z)
    ax.plot(x_hist[0, :], x_hist[1, :], x_hist[2, :], '-o', label='path')
    ax.plot(x_hist[0, -1], x_hist[1, -1], x_hist[2, -1], marker='^', markerfacecolor='yellow', markersize=12, label='final x')

    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.legend()

    fig = plt.figure()
    CS = plt.contour(X, Y, C, 10)
    plt.clabel(CS, inline=True, fontsize=10)

    plt.plot(x_hist[0, :], x_hist[1, :], '-o', label='path')
    plt.plot(x_hist[0, -1], x_hist[1, -1], marker='^', markerfacecolor='yellow', markersize=12, label='final x')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def plot_lp(f, x_hist):
    x0 = np.linspace(-0.1, 2.1, 1000)
    x1 = np.linspace(-0.1, 1.1, 1000)

    X, Y = np.meshgrid(x0, x1)
    C = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            C[i, j] = f.evaluate(np.array([[X[i, j]], [Y[i, j]]]))

    valid_indicies = np.ones(X.shape)
    X_plus_Y = X + Y
    valid_indicies[X_plus_Y < 1] = np.nan
    valid_indicies[Y > 1] = np.nan
    valid_indicies[X > 2] = np.nan
    valid_indicies[Y < 0] = np.nan

    plt.pcolor(X, Y, valid_indicies, label='feasible region')
    CS = plt.contour(X, Y, C, 30)
    plt.clabel(CS, inline=True, fontsize=10)

    plt.plot(x_hist[0, :], x_hist[1, :], '-o', label='path')
    plt.plot(x_hist[0, -1], x_hist[1, -1], marker='^', markerfacecolor='yellow', markersize=12, label='final x')

    plt.legend()
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title('LP convergence')
    plt.show()

