import torch
from copy import deepcopy
from components.tools.validation import evaluate_model


def train_and_save(n_episodes, train_env, val_env, agent):
    eps_start, eps_end, eps_decay = 1.0, 0.01, 0.995
    scores, balances, trades, best_metric, best_model = [], [], [], float('-inf'), None

    for i_episode in range(1, n_episodes + 1):
        state = train_env.reset()
        score, total_balance, total_trades = 0, train_env.balance, 0

        for t in range(1, n_episodes + 1):
            action = agent.act(state, eps=eps_start)
            next_state, reward, done, info = train_env.step(action)
            total_balance += reward
            if action != 0:  # Подсчет количества сделок (купля/продажа)
                total_trades += 1
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)
        balances.append(total_balance)
        trades.append(total_trades)
        eps_start = max(eps_end, eps_decay * eps_start)

        val_metric = evaluate_model(val_env, agent)

        if val_metric > best_metric:
            best_metric = val_metric
            best_model = deepcopy(agent.local_network.state_dict())
            torch.save(best_model, 'best_model.pth')
            print('Модель сохранена')

        print(f"Эпизод: {i_episode}/{n_episodes}, Награда: {score}, Баланс: {total_balance}, Сделок: {total_trades}")

    return scores, balances, trades, best_model
