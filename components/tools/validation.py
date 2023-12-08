def evaluate_model(val_env, agent):
    state = val_env.reset()
    initial_balance = val_env.balance
    while True:
        action = agent.act(state, eps=0.0)  # Используйте модель без исследования
        next_state, _, done, _ = val_env.step(action)
        state = next_state
        if done:
            break
    final_balance = val_env.balance
    return final_balance - initial_balance  # Возврат итогового результата
