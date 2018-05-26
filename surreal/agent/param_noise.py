import numpy as np
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