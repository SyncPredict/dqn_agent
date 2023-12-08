from decouple import AutoConfig, UndefinedValueError

config = AutoConfig(search_path=None)  # None означает использование стандартных путей

# Конфигурационные параметры
LEARNING_RATE = config('LEARNING_RATE', cast=float)
GAMMA = config('GAMMA', cast=float)
MEMORY_SIZE = config('MEMORY_SIZE', cast=int)
TAU = config('TAU', cast=float)
UPDATE_EVERY = config('UPDATE_EVERY', cast=int)
FILE_PATH = config('FILE_PATH', cast=str)
LOOKBACK_WINDOW_SIZE = config('LOOKBACK_WINDOW_SIZE', cast=int)
ACTION_SIZE = config('ACTION_SIZE', cast=int)
BATCH_SIZE = config('BATCH_SIZE', cast=int)
