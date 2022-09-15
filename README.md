# microdist 

**Note: this is still WIP, don't expect anything just yet.**

![look at those eyes](ray_puppy.jpg)

microdist is a tiny distributed deep learning framework.

It slightly modifies the ~150 lines of code of [micrograd](https://github.com/karpathy/micrograd),
by keeping the core engine untouched, and adding another ~100 LOC to the neural network layer
to introduce basic parameter servers using [Ray](https://docs.ray.io/en/latest).
The changes to the networks itself are just needed to get and send parameters to a server, resp. set them again after a training step.
We introduce one single dependency (Ray) and otherwise use plain Python, which all comes together at around ~250 LOC.
This might be useful for demo purposes or to teach the fundamentals of distributed training.

### Installation

```bash
git clone https://github.com/maxpumperla/microdist
cd microdist
python setup.py install
```

### Example usage

You can create some sample data, define a tiny `micrograd` model with two input neurons, two hidden layers of length 16 each and an output layer with a single neuron, and train this model distributed across 4 workers, communicating with a "master"
model via a parameter server by calling:

```python
from microdist import Model
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.1)
y = y*2 - 1

model = Model([2, 16, 16, 1])

model.dist_fit(X, y, steps=10, num_workers=4)
```

The notebook `demo.ipynb` provides a full demo of training an
2-layer neural network (MLP) binary classifier.


### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), 
which the tests use as a reference for verifying the correctness of the calculated
gradients. Then simply:

```bash
python -m pytest
```
