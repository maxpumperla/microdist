import ray
from microdist import Value, Model, ParameterServer

from sklearn.datasets import make_moons


def data():
    x, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1
    return x, y


def test_param_updates():

    ps = ParameterServer.remote(params=[Value(3), Value(2), Value(1)])
    ps.update_params.remote(grad=[2, 2, 2], lr=1)

    params = ray.get(ps.get_params.remote())
    assert params[0].data == 1


def test_dist_training_smoke():
    x, y = data()
    model = Model([2, 16, 16, 1])

    model.dist_fit(x, y, steps=10, num_workers=2)
    model.eval_step(x, y)
