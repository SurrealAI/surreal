import copy
import numpy as np
import torchx.nn as nnx

class ParameterNoise(object):
    def apply(self, model):
        pass

class NormalParameterNoise(ParameterNoise):
    def __init__(self, sigma):
        self.sigma = sigma
        print('Parameter noise initialized with sigma', self.sigma)

    def apply(self, params):
        # TODO: behavior of model.parameters() on networks with shared convolution
        for key in params:
            for k in params[key]:
                p = params[key][k]
                assert type(p) == np.ndarray
                shape = tuple(p.data.shape)
                noise = np.random.normal(0, self.sigma, size=shape)
                p = p + noise
                params[key][k] = p
        return params

    def __repr__(self):
        return 'NormalParameterNoise(sigma={})'.format(self.sigma)

class AdaptiveNormalParameterNoise(ParameterNoise):
    # Parameter noise adaptation based on https://arxiv.org/pdf/1706.01905.pdf
    def __init__(self, model_copy, module_dict_copy, target_stddev, compute_dist_interval=10, alpha=1.04, sigma=0.01):
        self.sigma = sigma
        self.target_stddev = target_stddev
        self.compute_dist_interval = compute_dist_interval
        self.alpha = alpha
        self.original_model = model_copy
        self.original_model_module_dict = module_dict_copy
        self.i = 0
        self.total_action_distance = 0.0
        print('Parameter noise initialized with sigma', self.sigma)

    def compute_action_distance(self, obs, modified_model_action):
        if self.i % self.compute_dist_interval == 0:
            # Calculate action, don't do forward pass on critic
            original_model_action, _ = self.original_model(obs, calculate_value=False)
            self.total_action_distance = (((original_model_action - modified_model_action) ** 2).sum()) ** 0.5
        self.i += 1

    def apply(self, params):
        # TODO: behavior of model.parameters() on networks with shared convolution
        if self.i > 0:
            mean_action_dist = self.total_action_distance / self.i
            print("Mean dist", mean_action_dist, "target", self.target_stddev, 'sigma', self.sigma)
            if mean_action_dist > self.target_stddev:
                self.sigma /= self.alpha
                print("Going down")
            else:
                self.sigma *= self.alpha
                print("Going up")
        self.i = 0
        # Deepcopy because module_dict converts params to tensor
        self.original_model_module_dict.load(copy.deepcopy(params))
        for key in params:
            for k in params[key]:
                p = params[key][k]
                assert type(p) == np.ndarray
                shape = tuple(p.shape)
                noise = np.random.normal(0, self.sigma, size=shape)
                p = p + noise
                params[key][k] = p
        return params

    def __repr__(self):
        return 'AdaptiveNormalParameterNoise(target={}, alpha={}, sigma={})'.format(self.target_distance, self.alpha, self.sigma)
