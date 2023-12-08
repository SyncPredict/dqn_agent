def execute_transaction(action, balance,  purchase_prices, current_price, transaction_fee_percent):
    """
    Выполнение транзакции на основе действия агента.
    """
    reward = 0

    if action == 1:  # Купить
        fee = current_price * transaction_fee_percent / 100
        balance -= current_price + fee
        purchase_prices.append(current_price)

    elif action == 2:  # Продать
        if purchase_prices:
            purchase_price = purchase_prices.pop(0)
            balance += current_price - (current_price * transaction_fee_percent / 100)
            reward = calculate_reward(action, [purchase_price], current_price, transaction_fee_percent)

    return balance, purchase_prices, reward

def calculate_reward(action, purchase_prices, current_price, transaction_fee_percent):
    """
    Расчет награды для действия агента.
    """
    reward = 0

    if action == 2:  # Продать
        if purchase_prices:
            purchase_price = purchase_prices[0]
            reward = current_price - purchase_price - (current_price * transaction_fee_percent / 100)

    return reward

