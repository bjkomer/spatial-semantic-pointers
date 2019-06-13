import nengo
import nengo.spa as spa
import numpy as np
from spatial_semantic_pointers.plots import SpatialHeatmap
from spatial_semantic_pointers.utils import make_good_unitary, get_heatmap_vectors, encode_point
import os

seed = 13
dim = 512
limit = 5
res = 128 #256
neurons_per_dim = 10

n_items = 5

rstate = np.random.RandomState(seed=13)

x_axis_sp = make_good_unitary(dim, rng=rstate)
y_axis_sp = make_good_unitary(dim, rng=rstate)

xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

vocab = spa.Vocabulary(dim)

vocab_vectors = np.zeros((n_items, dim))

print("Generating {0} vocab items".format(n_items))
for i in range(n_items):
    p = vocab.create_pointer()
    vocab_vectors[i, :] = p.v

fname = 'heatmap_vectors_{}items_dim{}_seed{}'.format(n_items, dim, seed)
if os.path.exists(fname):
    print("Loading heatmap vectors")
    data = np.load(fname)
    heatmap_vectors = data['heatmap_vectors']
    vocab_heatmap_vectors = data['vocab_heatmap_vectors']
else:
    print("Generating location heatmap vectors")
    heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)

    # Build set of heatmap vectors for visualizing the location of vocab items
    vocab_heatmap_vectors = np.zeros(
        (vocab_vectors.shape[0],
         heatmap_vectors.shape[0],
         heatmap_vectors.shape[1],
         heatmap_vectors.shape[2])
    )

    print("Generating vocab heatmap vectors")
    for vi in range(vocab_vectors.shape[0]):
        for xi in range(heatmap_vectors.shape[0]):
            for yi in range(heatmap_vectors.shape[1]):
                vocab_heatmap_vectors[vi, xi, yi, :] = (spa.SemanticPointer(data=heatmap_vectors[xi, yi]) * spa.SemanticPointer(data=vocab_vectors[vi])).v
    print("Generation Complete")

    np.savez(
        fname,
        heatmap_vectors=heatmap_vectors,
        vocab_heatmap_vectors=vocab_heatmap_vectors,
    )

spatial_heatmap = SpatialHeatmap(vocab_heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None)


class Environment(object):

    def __init__(self, vocab_vectors, x_axis_sp, y_axis_sp, n_items=5):

        self.n_items = n_items

        self.n_vocab = vocab_vectors.shape[0]
        self.dim = vocab_vectors.shape[1]

        self.vocab_vectors = vocab_vectors

        self.x_axis_sp = x_axis_sp
        self.y_axis_sp = y_axis_sp

    def __call__(self, t, x):

        output = spa.SemanticPointer(data=np.zeros(self.dim))
        for i in range(self.n_items):
            x_pos = x[i * 3]
            y_pos = x[i * 3 + 1]
            identity = np.clip(int(x[i * 3 + 2]), 0, self.n_vocab - 1)

            output += encode_point(x_pos, y_pos, self.x_axis_sp, self.y_axis_sp) * spa.SemanticPointer(data=self.vocab_vectors[identity])

        output.normalize()

        return output.v


model = spa.SPA(seed=seed)

with model:

    memory = nengo.Ensemble(
        n_neurons=dim*neurons_per_dim,
        dimensions=dim,
        neuron_type=nengo.Direct()  # setting to direct mode so it runs on small laptops
    )
    item_selector = nengo.Node([0])

    # Sliders to control the locations and identities of items
    identities = nengo.Node(list(np.arange(n_items)))
    locations = nengo.Node(list(np.random.uniform(low=-limit, high=limit, size=(n_items*2))))

    # environment where the spatial items exist
    env = nengo.Node(
        Environment(
            n_items=n_items,
            vocab_vectors=vocab_vectors,
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
        ),
        size_in=n_items*3,
        size_out=dim
    )

    heatmap_node = nengo.Node(
        spatial_heatmap,
        size_in=dim + 1, size_out=0,
    )

    # some hideous code for creating the index lists to line up connections
    location_indices = list(
        np.vstack([
            np.arange(0, n_items * 3 + 0, 3),
            np.arange(1, n_items * 3 + 1, 3)
        ]).T.flatten()
    )
    identity_indices = list(np.arange(2, n_items * 3 + 2, 3))

    nengo.Connection(locations, env[location_indices], synapse=None)
    nengo.Connection(identities, env[identity_indices], synapse=None)

    # feed the SSP from the environment into the memory
    nengo.Connection(env, memory)

    # item selector goes to the first index
    nengo.Connection(item_selector, heatmap_node[0])

    # the memory itself goes to all other indices
    nengo.Connection(memory, heatmap_node[1:])
