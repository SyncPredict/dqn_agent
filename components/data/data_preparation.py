from sklearn.model_selection import train_test_split
from components.data.processor import BitcoinDataProcessor

def prepare_data(file_path, lookback_window_size):
    bitcoin_data_processor = BitcoinDataProcessor(file_path)
    bitcoin_data = bitcoin_data_processor.data
    train_data, val_data = train_test_split(bitcoin_data, test_size=0.2, shuffle=False)
    state_size = len(train_data.columns) * lookback_window_size
    return train_data, val_data, state_size
