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

    x0 = np.linspace(0, 1, 1000)
    x1 = np.linspace(0, 1, 1000)

    X, Y = np.meshgrid(x0, x1)
    Z = np.maximum(0, 1 - X - Y)
    C = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            C[i, j] = f.evaluate(np.array([[X[i, j]], [Y[i, j]], [Z[i, j]]]))

    ax.contour3D(X, Y, C, 50)

    ax.scatter(x_hist[0, :], x_hist[1, :], x_hist[2, :], label='path')
    ax.scatter(x_hist[0, -1], x_hist[1, -1], x_hist[2, -1], s=30, marker='^', label='final x')

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, C, 50)
    #
    # x1_hist = []
    # x2_hist = []
    # z_hist = []
    # for x in x_hist.T:
    #     fx = f.evaluate(x)
    #     x1_hist.append(x[0])
    #     x2_hist.append(x[1])
    #     z_hist.append(fx)
    #
    # ax.plot(x1_hist, x2_hist, z_hist, 'o', markersize=12, label='path')
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('f(x)')
    # ax.legend()
    # colors = cm.Pastel1(range(x_hist.shape[1]))
    # fig, ax1 = plt.subplots()
    # CS = ax1.contour(X, Y, Z, 30)
    # ax1.clabel(CS, inline=True, fontsize=10)
    # for iteration, x in enumerate(x_hist.T):
    #     ax1.plot(x[0], x[1], 'o', color=colors[iteration], label='path'+str(iteration))

    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.legend()

    plt.show()
