import numpy as np
from common import args
from bandits import ContextualBandit, NeuralUCB


def hwithoutconstraints(action):
    wlist = np.load('rateListUsersYahoo.npy')[0] / max(np.load('rateListUsersYahoo.npy')[0]) / 4
    if type(action) is not np.ndarray:
        action = [i[0][0] for i in action]
    # print(action)
    gamma = 0.9
    currentw = wlist[action]
    p = 0
    for i in range(len(currentw)):
        tmp = (gamma**(i - 1)) * currentw[i]
        if i >= 1:
            for j in range(i - 1):
                tmp = tmp * (1 - currentw[j])
        p += tmp
    if type(p) is np.ndarray:
        p = p[0]
    p = np.clip(p, 0, 1)
    return np.random.choice([0, 1], p=[1 - p, p])


def main():
    T = int(5e3)
    n_arms = 10
    n_features = args.solDim * args.card
    noise_std = 0.5

    confidence_scaling_factor = noise_std

    n_sim = 1

    p = 0.2
    hidden_size = 4  # 16
    epochs = 100  # 100
    train_every = 10  # 10
    confidence_scaling_factor = 1.0
    use_cuda = False
    bandit = ContextualBandit(T, n_arms, n_features, hwithoutconstraints, noise_std=noise_std)

    # regrets = np.empty((n_sim, T))

    for i in range(n_sim):
        bandit.reset_rewards()
        model = NeuralUCB(bandit,
                          hidden_size=hidden_size,
                          reg_factor=1.0,
                          delta=0.1,
                          confidence_scaling_factor=confidence_scaling_factor,
                          training_window=100,
                          p=p,
                          learning_rate=0.01,
                          epochs=epochs,
                          train_every=train_every,
                          use_cuda=use_cuda
                          )

        model.run()
    pass


if __name__ == "__main__":
    main()
