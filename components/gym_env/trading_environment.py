import gym
import numpy as np
from gym import spaces
from components.gym_env.observation import get_next_observation
from components.gym_env.transaction import execute_transaction


class TradingEnvironment(gym.Env):
    """
    Торговая среда для Bitcoin, основанная на Gym.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, lookback_window_size=10):
        super(TradingEnvironment, self).__init__()

        self.df = df  # DataFrame с историческими данными
        self.max_steps = len(self.df) - 1
        self.current_step = 0

        self.lookback_window_size = lookback_window_size

        # Действия: 0 - удерживать, 1 - купить, 2 - продать
        self.action_space = spaces.Discrete(3)

        # Наблюдения: стоимость Bitcoin, объем торгов, рыночная капитализация и другие показатели
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(self.df.columns),), dtype=np.float32)

        # Инициализация состояния среды
        self.state = self._next_observation()
        self.balance = 10000  # Начальный баланс
        self.transaction_fee_percent = 0.1  # Пример: комиссия 0.1%
        self.purchase_prices = []  # Список цен покупки

    def _next_observation(self):
        """
        Получение наблюдения с окном исторических данных.
        """
        obs = get_next_observation(self.df, self.current_step, self.lookback_window_size)
        return obs

    def step(self, action):
        """
        Шаг среды на основе действия агента.
        """
        self.current_step += 1

        # Получаем текущую рыночную цену
        current_price = self.df.iloc[self.current_step]['rate']

        done = False

        balance, purchase_prices, reward = execute_transaction(action, self.balance, self.purchase_prices,
                                                               current_price, self.transaction_fee_percent)

        # Проверяем, не закончился ли эпизод
        if self.current_step >= self.max_steps:
            done = True

        # Обновляем состояние
        self.state = self._next_observation()

        # Рассчитываем награду

        return self.state, reward, done, {'current_price': current_price}

    def reset(self):
        """
        Сброс среды к начальному состоянию.
        """
        while len(self.purchase_prices) > 0:
            purchase_price = self.purchase_prices.pop(0)
            self.balance += (self.df.iloc[-1]['rate'] - purchase_price)

        self.current_step = 0
        self.state = self._next_observation()
        self.balance = 10000
        return self.state

    def render(self, mode='human', close=False):
        """
        Визуализация текущего состояния среды.
        """
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
