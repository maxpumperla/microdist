import ray
import random
import copy


from microdist import Value


def loss(model, x, y, batch_size=None, alpha=1e-4):
    if batch_size is None:
        x_b, y_b = x, y
    else:
        idx = random.choices(range(x.shape[0]), k=batch_size)
        x_b, y_b = x[idx], y[idx]
    inputs = [list(map(Value, x_row)) for x_row in x_b]

    # forward the model to get scores
    scores = list(map(model, inputs))

    # svm "max-margin" loss
    losses = [(1 + -y_i * score_i).relu() for y_i, score_i in zip(y_b, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))

    reg_loss = alpha * sum((p*p for p in model.get_params()))
    total_loss = data_loss + reg_loss

    accuracy = [(y_i > 0) == (score_i.data > 0) for y_i, score_i in zip(y_b, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


class Neuron:

    def __init__(self, nin, non_lin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.non_lin = non_lin

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.non_lin else act

    def set_params(self, params):
        self.w = params[:-1]
        self.b = params[-1]

    def get_params(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin, nout, **kwargs):
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def set_params(self, params):
        params = [params[i:i + self.nin + 1] for i in range(0, len(params), self.nin + 1)]
        for p, n in zip(params, self.neurons):
            n.set_params(p)

    def get_params(self):
        return [p for n in self.neurons for p in n.get_params()]


class Model:

    def __init__(self, sz):
        self.sizes = sz
        self.layers = [Layer(sz[i], sz[i+1], non_lin=i != len(sz) - 2) for i in range(len(sz) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def zero_grad(self):
        for p in self.get_params():
            p.grad = 0

    def set_params(self, params):
        sz = self.sizes
        for i, l in enumerate(self.layers):
            p, params = params[:sz[i+1]*(sz[i]+1)], params[sz[i+1]*(sz[i]+1):]
            l.set_params(p)

    def get_params(self):
        return [p for layer in self.layers for p in layer.get_params()]

    def eval_step(self, X, y, batch_size=None):

        total_loss, acc = loss(self, X, y, batch_size)

        self.zero_grad()
        total_loss.backward()

        print(f"Loss {total_loss.data}, accuracy {acc*100}%")

        return [p.grad for p in copy.deepcopy(self.get_params())]

    def fit(self, X, y, steps=100, batch_size=None):

        for step in range(steps):
            grad = self.eval_step(X, y, batch_size)
            lr = 1.0 - 0.9 * step / 100

            for p, g in zip(self.get_params(), grad):
                p.data -= lr * g

    def dist_fit(self, X, y, steps=10, batch_size=None, num_workers=10):

        ps = ParameterServer.remote(self.get_params())
        sizes = self.sizes

        @ray.remote
        def worker(param_server):

            worker_model = Model(sizes)

            for step in range(steps):
                ps_params = ray.get(param_server.get_params.remote())
                worker_model.set_params(ps_params)

                grad = worker_model.eval_step(X, y, batch_size)
                lr = 1.0 - 0.9 * step / 100

                param_server.update_params.remote(grad, lr)

        for _ in range(num_workers):
            worker.remote(ps)

        self.set_params(ray.get(ps.get_params.remote()))


@ray.remote
class ParameterServer:
    def __init__(self, params):
        self.params = params

    def get_params(self):
        return self.params

    def update_params(self, grad, lr):
        for p, g in zip(self.params, grad):
            p.data -= lr * g
