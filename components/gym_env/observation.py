import numpy as np

def get_next_observation(df, current_step, lookback_window_size):
    """
    Получение наблюдения с окном исторических данных.
    """
    if current_step < lookback_window_size:
        lookback_data = [df.iloc[0] for _ in range(lookback_window_size - current_step)] + \
                        [df.iloc[i] for i in range(current_step)]
    else:
        lookback_data = [df.iloc[current_step - i] for i in range(lookback_window_size)]

    obs = np.array([np.array(lookback_data).flatten()])
    return obs
