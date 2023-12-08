from components.tools.training_loop import train_and_save
from components.data.data_preparation import prepare_data
from components.tools.plot_results import plot_results
from components.gym_env.trading_environment import TradingEnvironment
from components.dqn.dqn_agent import DQNAgent
from components.tools.utils import FILE_PATH, LOOKBACK_WINDOW_SIZE, ACTION_SIZE

# Загрузка конфигураций

# Подготовка данных
train_data, val_data, STATE_SIZE = prepare_data(FILE_PATH, LOOKBACK_WINDOW_SIZE)

# Инициализация сред
train_env = TradingEnvironment(train_data, lookback_window_size=LOOKBACK_WINDOW_SIZE)
val_env = TradingEnvironment(val_data, lookback_window_size=LOOKBACK_WINDOW_SIZE)

# Создание экземпляра агента
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

# Параметры для обучения
n_episodes = 5

# Запуск цикла обучения
scores, balances, trades = train_and_save(n_episodes, train_env, val_env, agent)

# Визуализация результатов
plot_results(scores, balances, trades)
