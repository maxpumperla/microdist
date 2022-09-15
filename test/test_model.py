from microdist import Model, Layer, Neuron
import copy

from sklearn.datasets import make_moons


def data():
    x, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1
    return x, y


def test_neuron_params():
    neuron = Neuron(nin=12)
    params = neuron.get_params()

    neuron.set_params(params)
    assert params == neuron.get_params()


def test_layer_params():
    layer = Layer(nin=3, nout=2)
    params = layer.get_params()

    layer.set_params(params)
    assert params == layer.get_params()


def test_model_params():
    model = Model([2, 16, 32, 1])
    params = model.get_params()
    assert len(params) == 625

    model.set_params(params)

    assert model.get_params() == params


def test_model_train():
    x, y = data()
    model = Model([2, 16, 32, 1])
    params = copy.deepcopy(model.get_params())
    print(params[0])

    model.fit(x, y, 1)
    trained_params = model.get_params()
    print(trained_params[0])

    assert params != trained_params
    # TODO inspect model params


def test_dist_model_train():
    x, y = data()
    model = Model([2, 16, 32, 1])
    params = copy.deepcopy(model.get_params())

    model.dist_fit(x, y, 1)
    trained_params = model.get_params()

    assert params != trained_params
