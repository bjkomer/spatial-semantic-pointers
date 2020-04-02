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
# import nengo
import nengo_spa as spa
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, get_heatmap_vectors, ssp_to_loc_v, get_axes
from spatial_semantic_pointers.plots import plot_predictions_v
import matplotlib.pyplot as plt


class SpatialCleanup(object):

    def __init__(self, model_path, dim=512, hidden_size=512):

        self.model = FeedForward(dim=dim, hidden_size=hidden_size, output_size=dim)
        self.model.load_state_dict(torch.load(model_path), strict=True)

    def __call__(self, t, x):

        # Run SSP through the network, including conversions to and from pytorch tensors
        output = self.model(torch.Tensor(x).unsqueeze(0)).detach().numpy()
        mag = np.linalg.norm(output)
        if mag > 0:
            return output / np.linalg.norm(output)
        else:
            return output


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
        memory_sp = spa.SemanticPointer(data=np.zeros((dim,)))

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
                # item = spa.SemanticPointer(dim)
                item = spa.SemanticPointer(data=np.random.randn(dim)).normalized()
            else:
                item = spa.SemanticPointer(data=items_used[j])

            items[i, j, :] = item.v
            coords[i * n_items + j, 0] = x
            coords[i * n_items + j, 1] = y
            clean_ssps[i * n_items + j, :] = pos.v
            memory_sp += (pos * item)

        if normalize_memory:
            # memory_sp.normalize()
            memory_sp = memory_sp.normalized()

        memory[i, :] = memory_sp.v

        # Query for each item to get the noisy SSPs
        for j in range(n_items):
            noisy_ssps[i * n_items + j, :] = (memory_sp * ~spa.SemanticPointer(data=items[i, j, :])).v

    return clean_ssps, noisy_ssps, coords
    # return memory, items, coords, x_axis_sp, y_axis_sp


def main():
    parser = argparse.ArgumentParser('Train a network to clean up a noisy spatial semantic pointer')

    parser.add_argument('--loss', type=str, default='cosine', choices=['cosine', 'mse'])
    parser.add_argument('--noise-type', type=str, default='memory', choices=['memory', 'gaussian', 'both'])
    parser.add_argument('--sigma', type=float, default=1.0, help='sigma on the gaussian noise if noise-type==gaussian')
    parser.add_argument('--train-fraction', type=float, default=.8, help='proportion of the dataset to use for training')
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
    parser.add_argument('--use-hex-ssp', action='store_true')

    args = parser.parse_args()

    args.limits = tuple(float(v) for v in args.limits.split(','))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_name = 'data/ssp_cleanup_dataset_dim{}_seed{}_items{}_limit{}_samples{}.npz'.format(
        args.dim, args.seed, args.n_items, args.limits[1], args.n_samples
    )

    final_test_samples = 100
    final_test_items = 15
    final_test_dataset_name = 'data/ssp_cleanup_test_dataset_dim{}_seed{}_items{}_limit{}_samples{}.npz'.format(
        args.dim, args.seed, final_test_items, args.limits[1], final_test_samples
    )

    if not os.path.exists('data'):
        os.makedirs('data')

    rng = np.random.RandomState(seed=args.seed)
    if args.use_hex_ssp:
        x_axis_sp, y_axis_sp = get_axes(dim=args.dim, n=3, seed=args.seed)
    else:
        x_axis_sp = make_good_unitary(args.dim, rng=rng)
        y_axis_sp = make_good_unitary(args.dim, rng=rng)

    if args.noise_type == 'gaussian':
        # Simple generation
        clean_ssps = np.zeros((args.n_samples, args.dim))
        coords = np.zeros((args.n_samples, 2))
        for i in range(args.n_samples):
            x = np.random.uniform(low=args.limits[0], high=args.limits[1])
            y = np.random.uniform(low=args.limits[2], high=args.limits[3])

            clean_ssps[i, :] = encode_point(x, y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp).v
            coords[i, 0] = x
            coords[i, 1] = y
        # Gaussian noise will be added later
        noisy_ssps = clean_ssps.copy()
    else:

        if os.path.exists(dataset_name):
            print("Loading dataset")
            data = np.load(dataset_name)
            clean_ssps = data['clean_ssps']
            noisy_ssps = data['noisy_ssps']
        else:
            print("Generating SSP cleanup dataset")
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

    # check if the final test set has been generated yet
    if os.path.exists(final_test_dataset_name):
        print("Loading final test dataset")
        final_test_data = np.load(final_test_dataset_name)
        final_test_clean_ssps = final_test_data['clean_ssps']
        final_test_noisy_ssps = final_test_data['noisy_ssps']
    else:
        print("Generating final test dataset")
        final_test_clean_ssps, final_test_noisy_ssps, final_test_coords = generate_cleanup_dataset(
            x_axis_sp=x_axis_sp,
            y_axis_sp=y_axis_sp,
            n_samples=final_test_samples,
            dim=args.dim,
            n_items=final_test_items,
            limits=args.limits,
            seed=args.seed,
        )
        print("Final test generation complete. Saving dataset")
        np.savez(
            final_test_dataset_name,
            clean_ssps=final_test_clean_ssps,
            noisy_ssps=final_test_noisy_ssps,
            coords=final_test_coords,
            x_axis_vec=x_axis_sp.v,
            y_axis_vec=x_axis_sp.v,
        )

    # Add gaussian noise if required
    if args.noise_type == 'gaussian' or args.noise_type == 'both':
        noisy_ssps += np.random.normal(loc=0, scale=args.sigma, size=noisy_ssps.shape)

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
    dataset_final_test = CoordDecodeDataset(vectors=final_test_noisy_ssps, coords=final_test_clean_ssps)

    trainloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    # For testing just do everything in one giant batch
    testloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
    )

    final_testloader = torch.utils.data.DataLoader(
        dataset_final_test, batch_size=len(dataset_final_test), shuffle=False, num_workers=0,
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

            avg_cosine_loss += cosine_loss.data.item()
            avg_mse_loss += mse_loss.data.item()
            n_batches += 1

            if args.loss == 'cosine':
                cosine_loss.backward()
            else:
                mse_loss.backward()

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

        for label, loader in zip(['test', 'final_test'], [testloader, final_testloader]):

            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(loader):

                noisy, clean = data

                outputs = model(noisy)

                mse_loss = mse_criterion(outputs, clean)
                # Modified to use CosineEmbeddingLoss
                cosine_loss = cosine_criterion(outputs, clean, torch.ones(len(loader)))

                print(cosine_loss.data.item())

            if args.logdir != '':
                # TODO: get a visualization of the performance

                # show plots of the noisy, clean, and cleaned up with the network
                # note that the plotting mechanism itself uses nearest neighbors, so has a form of cleanup built in

                xs = np.linspace(args.limits[0], args.limits[1], 256)
                ys = np.linspace(args.limits[0], args.limits[1], 256)

                heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)

                noisy_coord = ssp_to_loc_v(
                    noisy,
                    heatmap_vectors, xs, ys
                )

                pred_coord = ssp_to_loc_v(
                    outputs,
                    heatmap_vectors, xs, ys
                )

                clean_coord = ssp_to_loc_v(
                    clean,
                    heatmap_vectors, xs, ys
                )

                fig_noisy_coord, ax_noisy_coord = plt.subplots()
                fig_pred_coord, ax_pred_coord = plt.subplots()
                fig_clean_coord, ax_clean_coord = plt.subplots()

                plot_predictions_v(
                    noisy_coord,
                    clean_coord,
                    ax_noisy_coord, min_val=args.limits[0], max_val=args.limits[1], fixed_axes=True
                )

                plot_predictions_v(
                    pred_coord,
                    clean_coord,
                    ax_pred_coord, min_val=args.limits[0], max_val=args.limits[1], fixed_axes=True
                )

                plot_predictions_v(
                    clean_coord,
                    clean_coord,
                    ax_clean_coord, min_val=args.limits[0], max_val=args.limits[1], fixed_axes=True
                )

                writer.add_figure('{}/original_noise'.format(label), fig_noisy_coord)
                writer.add_figure('{}/test_set_cleanup'.format(label), fig_pred_coord)
                writer.add_figure('{}/ground_truth'.format(label), fig_clean_coord)
                # fig_hist = plot_histogram(predictions=outputs, coords=coord)
                # writer.add_figure('test set histogram', fig_hist)
                writer.add_scalar('{}/test_cosine_loss'.format(label), cosine_loss.data.item())
                writer.add_scalar('{}/test_mse_loss'.format(label), mse_loss.data.item())

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
