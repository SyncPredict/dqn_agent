import numpy as np
from matplotlib import pyplot as plt


def plot_results(scores, balances, trades):
    # Вычисление скользящего среднего
    window_size = 50  # Размер окна для скользящего среднего
    moving_avg = [np.mean(scores[max(0, i-window_size):(i+1)]) for i in range(len(scores))]

    # График накопленных наград (прибыли) по эпизодам
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores, label='Награды за эпизод')
    plt.plot(moving_avg, label='Скользящее среднее', color='orange')
    plt.title('Награды по эпизодам')
    plt.xlabel('Эпизод')
    plt.ylabel('Награда')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(balances, label='Баланс')
    plt.title('Баланс по эпизодам')
    plt.xlabel('Эпизод')
    plt.ylabel('Баланс')
    plt.legend()

    plt.subplot(1, 3, 3)  # Добавление третьего подграфика
    plt.plot(trades, label='Количество сделок')
    plt.title('Количество сделок по эпизодам')
    plt.xlabel('Эпизод')
    plt.ylabel('Сделки')
    plt.legend()

    plt.show()
