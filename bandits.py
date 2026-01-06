import abc
import numpy as np
import torch
import torch.nn as nn
import gurobipy

from common import args
from network import Model
from utils import checkFea, onehot_to_binaryvec_withoutpos, onehot_to_rank, inv_sherman_morrison, posdata_to_onehot
from solver import solver, solver_quad


class UCB(abc.ABC):
    """Base class for UBC methods.
    """

    def __init__(self,
                 bandit,
                 reg_factor=1.0,
                 confidence_scaling_factor=-1.0,
                 delta=0.1,
                 train_every=1,
                 throttle=int(1e2),
                 ):
        # bandit object, contains features and generated rewards
        self.bandit = bandit
        # L2 regularization strength
        self.reg_factor = reg_factor
        # Confidence bound with probability 1-delta
        self.delta = delta
        # multiplier for the confidence bound (default is bandit reward noise std dev)
        if confidence_scaling_factor == -1.0:
            confidence_scaling_factor = bandit.noise_std
        self.confidence_scaling_factor = confidence_scaling_factor

        # train approximator only every few rounds
        self.train_every = train_every
        self.best_recommended_reward = -200
        self.best_recommended_action = np.array([1] * args.card + [0] * (self.bandit.n_features - args.card))
        self.rewards_list = []
        self.best_sample_rewards_list = []
        self.feasi = []
        # throttle tqdm updates
        self.throttle = throttle
        self.solverSol = None
        self.reset()

    def reset_upper_confidence_bounds(self):
        """Initialize upper confidence bounds and related quantities.
        """
        self.exploration_bonus = np.empty((self.bandit.n_arms))
        self.mu_hat = np.empty((self.bandit.n_arms))
        self.meta_mu_hat = np.empty((self.bandit.n_features))
        self.cross_mu_hat = np.empty((self.bandit.n_features**2))
        self.upper_confidence_bounds = np.ones((self.bandit.n_arms))
        self.meta_values = np.ones((self.bandit.n_features))
        self.cross_values = np.ones((self.bandit.n_features**2))
        self.knn_metabest = -100
        self.best_rewards_oracle = -100

    def reset_actions(self):
        """Initialize cache of actions.
        """
        self.actions = np.empty(self.bandit.T).astype('int')

    def reset_A_inv(self):
        """Initialize n_arms square matrices representing the inverses
        of exploration bonus matrices.
        """
        self.A_inv = np.array(
            [
                np.eye(self.approximator_dim) / self.reg_factor for _ in self.bandit.arms
            ]
        )

    def reset_grad_approx(self):
        """Initialize the gradient of the approximator w.r.t its parameters.
        """
        self.grad_approx = np.zeros((self.bandit.n_arms, self.approximator_dim))

    def sample_action(self):
        """Return the action to play based on current estimates
        """
        if self.iteration >= max(self.bandit.n_features * 2, 100):
            return int(torch.argmax(self.upper_confidence_bounds).item())
        else:
            return np.random.choice(self.bandit.n_arms)

    @abc.abstractmethod
    def reset(self):
        """Initialize variables of interest.
        To be defined in children classes.
        """
        pass

    @property
    @abc.abstractmethod
    def approximator_dim(self):
        """Number of parameters used in the approximator.
        """
        pass

    @property
    @abc.abstractmethod
    def confidence_multiplier(self):
        """Multiplier for the confidence exploration bonus.
        To be defined in children classes.
        """
        pass

    # @abc.abstractmethod
    # def update_confidence_bounds(self):
    #     """Update the confidence bounds for all arms at time t.
    #     To be defined in children classes.
    #     """
    #     pass
    #
    @abc.abstractmethod
    def update_output_gradient(self):
        """Compute output gradient of the approximator w.r.t its parameters.
        """
        pass

    @abc.abstractmethod
    def train(self):
        """Update approximator.
        To be defined in children classes.
        """
        pass

    @abc.abstractmethod
    def predict(self):
        """Predict rewards based on an approximator.
        To be defined in children classes.
        """
        pass

    def update_confidence_bounds(self):
        """Update confidence bounds and related quantities for all arms.
        """
        if self.iteration > max(self.bandit.n_features * 2, 100):  # self.bandit.n_features:
            self.bandit.features[self.iteration][2] = self.best_recommended_action
            self.bandit.features[self.iteration][0] = self.solverSol
            self.bandit.features[self.iteration][1] = self.solverSol_quad
            if len(self.elite) >= 5:
                for kk in range(min(len(self.elite), self.bandit.n_arms // 2)):
                    self.bandit.features[self.iteration][-kk] = self.elite[kk]
        self.update_output_gradient()

        # UCB exploration bonus
        self.exploration_bonus = np.array(
            [
                -10. * checkFea(onehot_to_binaryvec_withoutpos(self.bandit.features[self.iteration][a] * np.sqrt(args.card))) + 0.01 * np.sqrt(np.dot(self.grad_approx[a], np.dot(self.A_inv[a], self.grad_approx[a].T))) for a in self.bandit.arms
            ]
        )
        self.model.eval()
        self.mu_hat = self.model.forward(
            torch.FloatTensor(self.bandit.features[self.iteration]).to(self.device)
        ).detach().squeeze()
        # print(self.bandit.features[self.iteration][0][:10],self.bandit.features[self.iteration][-1][:10])
        self.meta_mu_hat = self.model.forward(
            torch.FloatTensor(self.bandit.meta_features[self.iteration]).to(self.device)
        ).detach().squeeze()
        if self.iteration >= max(self.bandit.n_features * 2, 200):
            self.cross_mu_hat = self.model.forward(
                torch.FloatTensor(self.bandit.cross_features).to(self.device)
            ).detach().squeeze()
        self.bandit.rewards[self.iteration] = np.array([self.bandit.hwithoutconstraints(onehot_to_rank(np.sqrt(args.card) * self.bandit.features[self.iteration, k])) for k in range(self.bandit.n_arms)])
        # estimated combined bound for reward
        self.meta_values = self.meta_mu_hat  # + self.meta_exploration_bonus[self.iteration]
        if self.iteration >= max(self.bandit.n_features * 2, 100):
            self.cross_values = self.cross_mu_hat
            for i in range(self.bandit.n_features):
                for j in range(self.bandit.n_features):
                    if i == j:
                        self.cross_values[i * self.bandit.n_features + j] = self.meta_mu_hat[i]
                    else:
                        self.cross_values[i * self.bandit.n_features + j] = (self.cross_mu_hat[i * self.bandit.n_features + j] - self.meta_mu_hat[i] - self.meta_mu_hat[j]) / 2
            self.cross_values = self.cross_values.reshape(self.bandit.n_features, self.bandit.n_features)
        with gurobipy.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('IterationLimit', 600)
            env.start()
            with gurobipy.Model(env=env) as m:
                self.solverSol = np.divide(solver(self.meta_values, m), np.sqrt(args.card))
        if self.iteration >= max(self.bandit.n_features * 2, 100):
            with gurobipy.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.setParam('IterationLimit', 600)
                env.start()
                with gurobipy.Model(env=env) as m:
                    self.solverSol_quad = np.divide(solver_quad(self.cross_values, m), np.sqrt(args.card))

        # estimated combined bound for reward
        self.upper_confidence_bounds = self.mu_hat + self.exploration_bonus
        #   self.knn_metabest=np.divide(posdata_to_onehot(networkOutput_to_posdata(self.meta_values)).detach().numpy(),np.sqrt(args.card))
        #  self.bandit.knn_metabestValue =self.bandit.hwithoutconstraints(onehot_to_rank(np.sqrt(args.card)*self.knn_metabest))
        # self.bandit.best_rewards_oracle =max( max(np.max(self.bandit.rewards, axis=1)),self.bandit.knn_metabestValue)
        # self.best_rewards_oracle=max(max(self.best_rewards_oracle,self.bandit.best_rewards_oracle),self.bandit.hwithoutconstraints(onehot_to_rank(np.sqrt(args.card)*self.knn_metabest)))

    def update_A_inv(self):
        self.A_inv[self.action] = inv_sherman_morrison(
            self.grad_approx[self.action],
            self.A_inv[self.action]
        )


class ContextualBandit():

    def __init__(self,
                 T,
                 n_arms,
                 n_features,
                 hwithoutconstraints,
                 noise_std=1.0,
                 ):
        # number of rounds
        # number of rounds
        self.T = T
        # number of arms
        self.n_arms = n_arms
        # number of features for each arm
        self.n_features = n_features
        # average reward function
        # h : R^d -> R
        self.hwithoutconstraints = hwithoutconstraints
        self.knn_metabestValue = -100
        self.best_rewards_oracle = -100
        # standard deviation of Gaussian reward noise
        self.noise_std = noise_std
        # generate random features
        self.reset()

    @property
    def arms(self):
        """Return [0, ...,n_arms-1]
        """
        return range(self.n_arms)

    def reset(self):
        """Generate new features and new rewards.
        """
        self.reset_features()
        self.reset_rewards()

    def reset_features(self):
        """Generate normalized random N(0,1) features.
        """
        x1, x2 = [], []
        for i in range(self.T):
            tmp1, tmp2 = [], []
            for j in range(self.n_arms):

                a = np.array([0] * (args.solDim - args.card) + [1] * args.card)
                np.random.shuffle(a)
                rank = np.argwhere(a > 0.5)
                a = np.divide(posdata_to_onehot(rank), np.sqrt(args.card))
                tmp1.append(a)
            for j in range(self.n_features):
                a = np.array([0.0] * j + [1.0] + [0.0] * (self.n_features - j - 1))
                tmp2.append(a)
            x1.append(tmp1)
            x2.append(tmp2)
        x1, x2 = np.array(x1), np.array(x2)
        # x1 /= np.repeat(np.linalg.norm(x1, axis=-1, ord=2), self.n_features).reshape(self.T, self.n_arms, self.n_features)
        self.features = x1
        self.meta_features = x2
        self.cross_features = []
        for i in range(self.n_features):
            for j in range(self.n_features):
                tmp = np.zeros(self.n_features)
                tmp[i] = 1
                tmp[j] = 1
                self.cross_features.append(tmp)

    def reset_rewards(self):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """

        self.rewards = np.random.random([self.T, self.n_arms])
        # to be used only to compute regret, NOT by the algorithm itself
        self.best_rewards_oracle = max(max(np.max(self.rewards, axis=1)), self.knn_metabestValue)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)


class NeuralUCB(UCB):
    """Neural UCB.
    """

    def __init__(self,
                 bandit,
                 hidden_size=20,
                 n_layers=2,
                 reg_factor=1.0,
                 delta=0.01,
                 confidence_scaling_factor=-1.0,
                 training_window=100,
                 p=0.0,
                 learning_rate=0.01,
                 epochs=1,
                 train_every=1,
                 throttle=1,
                 use_cuda=False,
                 ):

        # hidden size of the NN layers
        self.hidden_size = hidden_size
        # number of layers
        self.n_layers = n_layers

        # number of rewards in the training buffer
        self.training_window = training_window

        # NN parameters
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.use_cuda = use_cuda
        if self.use_cuda:
            raise Exception(
                'Not yet CUDA compatible : TODO for later (not necessary to obtain good results')
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        # dropout rate
        self.p = p

        # neural network
        self.model = Model(input_size=bandit.n_features,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           p=self.p
                           ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        super().__init__(bandit,
                         reg_factor=reg_factor,
                         confidence_scaling_factor=confidence_scaling_factor,
                         delta=delta,
                         throttle=throttle,
                         train_every=train_every,
                         )

    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

    @property
    def confidence_multiplier(self):
        """Constant equal to confidence_scaling_factor
        """
        return self.confidence_scaling_factor

    def update_output_gradient(self):
        """Get gradient of network prediction w.r.t network weights.
        """
        for a in self.bandit.arms:
            x = torch.FloatTensor(
                self.bandit.features[self.iteration, a].reshape(1, -1)
            ).to(self.device)

            self.model.zero_grad()
            y = self.model(x)
            y.backward()

            self.grad_approx[a] = torch.cat(
                [w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.model.parameters() if w.requires_grad]
            ).to(self.device)

    def reset(self):
        """Reset the internal estimates.
        """
        self.reset_upper_confidence_bounds()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()
        self.iteration = 0

    def train(self):
        """Train neural approximator.
        """
        iterations_so_far = range(np.max([0, self.iteration - self.training_window]), self.iteration + 1)
        actions_so_far = self.actions[np.max([0, self.iteration - self.training_window]):self.iteration + 1]

        x_train = torch.FloatTensor(self.bandit.features[iterations_so_far, actions_so_far]).to(self.device)
        y_train = torch.FloatTensor(self.bandit.rewards[iterations_so_far, actions_so_far]).squeeze().to(self.device)

        # train mode
        self.model.train()
        for _ in range(self.epochs):
            y_pred = self.model.forward(x_train).squeeze()
            loss = nn.MSELoss()(y_train, y_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self):
        """Predict reward.
        """
        # eval mode
        self.model.eval()
        self.mu_hat = self.model.forward(
            torch.FloatTensor(self.bandit.features[self.iteration]).to(self.device)
        ).detach().squeeze()
