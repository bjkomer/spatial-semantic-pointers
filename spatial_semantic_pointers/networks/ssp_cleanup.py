# Given a 'noisy' spatial semantic pointer that is the result of a query, clean it up to the best pure SSP
# One use of this function is a component in sliding single objects in a memory
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tensorboardX import SummaryWriter
import argparse
import json
from datetime import datetime
import os.path as osp
import os
import nengo
from spatial_semantic_pointers.utils import encode_point, make_good_unitary


class SpatialCleanup(object):

    def __init__(self, model_path, dim=512, hidden_size=512):

        self.model = FeedForward(dim=dim, hidden_size=hidden_size, output_size=dim)
        self.model.load_state_dict(torch.load(model_path), strict=True)

    def __call__(self, x, t):

        # Run SSP through the network, including conversions to and from pytorch tensors
        return self.model(torch.Tensor(x).unsqueeze(0)).detach().numpy()[0, :]


class CoordDecodeDataset(data.Dataset):

    def __init__(self, vectors, coords):

        self.vectors = vectors.astype(np.float32)
        self.coords = coords.astype(np.float32)

        self.n_vectors = self.vectors.shape[0]
        self.dim = self.vectors.shape[1]

    def __getitem__(self, index):
        return self.vectors[index], self.coords[index]

    def __len__(self):
        return self.n_vectors


class FeedForward(nn.Module):
    """
    Takes in a continuous semantic pointer and computes some target from it
    """

    def __init__(self, dim=512, hidden_size=512, output_size=2):
        super(FeedForward, self).__init__()

        # dimensionality of the semantic pointers
        self.dim = dim

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer = nn.Linear(self.dim, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):

        features = F.relu(self.input_layer(inputs))
        prediction = self.output_layer(features)

        return prediction


def generate_cleanup_dataset(
        x_axis_sp,
        y_axis_sp,
        n_samples,
        dim,
        n_items,
        item_set=None,
        allow_duplicate_items=False,
        limits=(-1, 1, -1, 1),
        seed=13,
        normalize_memory=True):
    """
    TODO: fix this description
    Create a dataset of memories that contain items bound to coordinates

    :param n_samples: number of memories to create
    :param dim: dimensionality of the memories
    :param n_items: number of items in each memory
    :param item_set: optional list of possible item vectors. If not supplied they will be generated randomly
    :param allow_duplicate_items: if an item set is given, this will allow the same item to be at multiple places
    :param x_axis_sp: optional x_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param y_axis_sp: optional y_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param limits: limits of the 2D space (x_low, x_high, y_low, y_high)
    :param seed: random seed for the memories and axis vectors if not supplied
    :param normalize_memory: if true, call normalize() on the memory semantic pointer after construction
    :return: memory, items, coords, x_axis_sp, y_axis_sp, z_axis_sp
    """
    # This seed must match the one that was used to generate the model
    np.random.seed(seed)

    # Memory containing n_items of items bound to coordinates
    memory = np.zeros((n_samples, dim))

    # SP for the item of interest
    items = np.zeros((n_samples, n_items, dim))

    # Coordinate for the item of interest
    coords = np.zeros((n_samples * n_items, 2))

    # Clean ground truth SSP
    clean_ssps = np.zeros((n_samples * n_items, dim))

    # Noisy output SSP
    noisy_ssps = np.zeros((n_samples * n_items, dim))

    for i in range(n_samples):
        memory_sp = nengo.spa.SemanticPointer(data=np.zeros((dim,)))

        # If a set of items is given, choose a subset to use now
        if item_set is not None:
            items_used = np.random.choice(item_set, size=n_items, replace=allow_duplicate_items)
        else:
            items_used = None

        for j in range(n_items):

            x = np.random.uniform(low=limits[0], high=limits[1])
            y = np.random.uniform(low=limits[2], high=limits[3])

            pos = encode_point(x, y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)

            if items_used is None:
                item = nengo.spa.SemanticPointer(dim)
            else:
                item = nengo.spa.SemanticPointer(data=items_used[j])

            items[i, j, :] = item.v
            coords[i * n_items + j, 0] = x
            coords[i * n_items + j, 1] = y
            clean_ssps[i * n_items + j, :] = pos.v
            memory_sp += (pos * item)

        if normalize_memory:
            memory_sp.normalize()

        memory[i, :] = memory_sp.v

        # Query for each item to get the noisy SSPs
        for j in range(n_items):
            noisy_ssps[i * n_items + j, :] = (memory_sp * ~nengo.spa.SemanticPointer(data=items[i, j, :])).v

    return clean_ssps, noisy_ssps, coords
    # return memory, items, coords, x_axis_sp, y_axis_sp


def main():
    parser = argparse.ArgumentParser('Train a network to clean up a noisy spatial semantic pointer')

    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--train-fraction', type=float, default=.5, help='proportion of the dataset to use for training')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of memories to generate. Total samples will be n-samples * n-items')
    parser.add_argument('--n-items', type=int, default=12, help='number of items in memory. Proxy for noisiness')
    parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the semantic pointers')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size of the cleanup network')
    parser.add_argument('--limits', type=str, default="-5,5,-5,5", help='The limits of the space')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--logdir', type=str, default='ssp_cleanup',
                        help='Directory for saved model and tensorboard log')
    parser.add_argument('--load-model', type=str, default='', help='Optional model to continue training from')
    parser.add_argument('--name', type=str, default='',
                        help='Name of output folder within logdir. Will use current date and time if blank')
    parser.add_argument('--weight-histogram', action='store_true', help='Save histograms of the weights if set')

    args = parser.parse_args()

    args.limits = tuple(float(v) for v in args.limits.split(','))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_name = 'data/ssp_cleanup_dataset_dim{}_seed{}_items{}_samples{}.npz'.format(
        args.dim, args.seed, args.n_items, args.n_samples
    )

    if not os.path.exists('data'):
        os.makedirs('data')

    rng = np.random.RandomState(seed=args.seed)
    x_axis_sp = make_good_unitary(args.dim, rng=rng)
    y_axis_sp = make_good_unitary(args.dim, rng=rng)

    if os.path.exists(dataset_name):
        print("Loading dataset")
        data = np.load(dataset_name)
        clean_ssps = data['clean_ssps']
        noisy_ssps = data['noisy_ssps']
    else:
        print("Generating SSP cleanup dataset")
        # TODO: save the dataset the first time it is created, so it can be loaded the next time
        clean_ssps, noisy_ssps, coords = generate_cleanup_dataset(
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
            n_samples=args.n_samples,
            dim=args.dim,
            n_items=args.n_items,
            limits=args.limits,
            seed=args.seed,
        )
        print("Dataset generation complete. Saving dataset")
        np.savez(
            dataset_name,
            clean_ssps=clean_ssps,
            noisy_ssps=noisy_ssps,
            coords=coords,
            x_axis_vec=x_axis_sp.v,
            y_axis_vec=x_axis_sp.v,
        )

    n_samples = clean_ssps.shape[0]
    n_train = int(args.train_fraction * n_samples)
    n_test = n_samples - n_train
    assert(n_train > 0 and n_test > 0)
    train_clean = clean_ssps[:n_train, :]
    train_noisy = noisy_ssps[:n_train, :]
    test_clean = clean_ssps[n_train:, :]
    test_noisy = noisy_ssps[n_train:, :]

    # NOTE: this dataset is actually generic and can take any input/output mapping
    dataset_train = CoordDecodeDataset(vectors=train_noisy, coords=train_clean)
    dataset_test = CoordDecodeDataset(vectors=test_noisy, coords=test_clean)

    trainloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    # For testing just do everything in one giant batch
    testloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
    )

    model = FeedForward(dim=dataset_train.dim, hidden_size=args.hidden_size, output_size=dataset_train.dim)

    # Open a tensorboard writer if a logging directory is given
    if args.logdir != '':
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        save_dir = osp.join(args.logdir, current_time)
        writer = SummaryWriter(log_dir=save_dir)
        if args.weight_histogram:
            # Log the initial parameters
            for name, param in model.named_parameters():
                writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), 0)

    mse_criterion = nn.MSELoss()
    cosine_criterion = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for e in range(args.epochs):
        print('Epoch: {0}'.format(e + 1))

        avg_mse_loss = 0
        avg_cosine_loss = 0
        n_batches = 0
        for i, data in enumerate(trainloader):

            noisy, clean = data

            if noisy.size()[0] != args.batch_size:
                continue  # Drop data, not enough for a batch
            optimizer.zero_grad()

            outputs = model(noisy)

            mse_loss = mse_criterion(outputs, clean)
            # Modified to use CosineEmbeddingLoss
            cosine_loss = cosine_criterion(outputs, clean, torch.ones(args.batch_size))
            # print(loss.data.item())
            avg_cosine_loss += cosine_loss.data.item()
            avg_mse_loss += mse_loss.data.item()
            n_batches += 1

            cosine_loss.backward()

            # print(loss.data.item())

            optimizer.step()

        print(avg_cosine_loss / n_batches)

        if args.logdir != '':
            if n_batches > 0:
                avg_cosine_loss /= n_batches
                writer.add_scalar('avg_cosine_loss', avg_cosine_loss, e + 1)
                writer.add_scalar('avg_mse_loss', avg_mse_loss, e + 1)

            if args.weight_histogram and (e + 1) % 10 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)

    print("Testing")
    with torch.no_grad():

        # Everything is in one batch, so this loop will only happen once
        for i, data in enumerate(testloader):

            noisy, clean = data

            outputs = model(noisy)

            mse_loss = mse_criterion(outputs, clean)
            # Modified to use CosineEmbeddingLoss
            cosine_loss = cosine_criterion(outputs, clean, torch.ones(len(dataset_test)))

            print(cosine_loss.data.item())

        if args.logdir != '':
            # TODO: get a visualization of the performance
            # fig_pred = plot_predictions(
            #     predictions=outputs, coords=coord,
            #     min_val=min(args.limits[0], args.limits[2]),
            #     max_val=max(args.limits[1], args.limits[3])
            # )
            # writer.add_figure('test set predictions', fig_pred)
            # fig_truth = plot_predictions(
            #     predictions=coord, coords=coord,
            #     min_val=min(args.limits[0], args.limits[2]),
            #     max_val=max(args.limits[1], args.limits[3])
            # )
            # writer.add_figure('ground truth', fig_truth)
            # fig_hist = plot_histogram(predictions=outputs, coords=coord)
            # writer.add_figure('test set histogram', fig_hist)
            writer.add_scalar('test_cosine_loss', cosine_loss.data.item())
            writer.add_scalar('test_mse_loss', mse_loss.data.item())

    # Close tensorboard writer
    if args.logdir != '':
        writer.close()

        torch.save(model.state_dict(), osp.join(save_dir, 'model.pt'))

        params = vars(args)
        # # Additionally save the axis vectors used
        # params['x_axis_vec'] = list(x_axis_sp.v)
        # params['y_axis_vec'] = list(y_axis_sp.v)
        with open(osp.join(save_dir, "params.json"), "w") as f:
            json.dump(params, f)


if __name__ == '__main__':
    main()
