from numpy.random import RandomState
from numpy import dot, prod, zeros, sign, average, seterr, exp
from numpy.linalg import norm


seterr(all='raise')


class LTFArray:

    def __init__(self, n, k, random_state=0):
        self.k = k
        self.n = n
        self.weights = RandomState(seed=random_state).normal(0, 1, (k, n))

    def val(self, x, js=None):
        if js is None:
            js = range(self.k)
        ret = prod([dot(self.weights[j], x) for j in js])
        #ret = prod([
        #    sum([ self.weights[j, i] if x[i] == 1 else -self.weights[j, i] for i in range(self.n)])
        #    for j in js ])
        ret = min(ret, 50)
        ret = max(ret, -50)
        return ret


def learn(n, k, observations, model=None):

    if model is None:
        model = LTFArray(n, k, random_state=1)

    for _ in range(1):
        # compute gradient
        grad = zeros((k, n))
        for j in range(k):
            for i in range(n):
                # compute partial derivative
                for (x, r) in observations:
                    grad[j, i] += \
                        (1 - (1/(1+exp(-r*model.val(x))))) * \
                        r * \
                        model.val(x, js=set(range(k)) - {j}) * \
                        x[i]

        model.weights += grad / norm(grad)

    return model


def accuracy(a, b, N=1000):
    assert a.n == b.n
    n = a.n
    return .5 * average([sign(a.val(c) * b.val(c)) for c in RandomState(seed=4).choice((-1, 1), (N, n))]) + .5


def simulate_and_learn(n, k, N, random_state=3):
    simulation = LTFArray(n, k)
    challenges = RandomState(seed=random_state).choice((-1, 1), (N, n))
    observations = [ (c, sign(simulation.val(c))) for c in challenges ]
    model = None
    acc = 0
    for _ in range(20):
        model = learn(n, k, observations, model)
        acc = accuracy(simulation, model)
        print('accuracy: %.5f' % acc)
    return acc


print('final accuracy: %f' % simulate_and_learn(64, 2, 10000))
