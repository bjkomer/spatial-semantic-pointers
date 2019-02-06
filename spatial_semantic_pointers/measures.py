import numpy as np


def item_match_exact(sp, vocab_vectors, item, sim_threshold=0.0):
    if sp.__class__.__name__ == 'SemanticPointer':
        sim = np.tensordot(sp.v, vocab_vectors, axes=([0], [1]))
    else:
        sim = np.tensordot(sp, vocab_vectors, axes=([0], [1]))

    ind = np.argmax(sim)

    if sim[ind] < sim_threshold:
        return 0

    if np.allclose(vocab_vectors[ind], item):
        return 1
    else:
        return 0


def item_match(sp, vocab_vectors, item, sim_threshold=0.0):
    if sp.__class__.__name__ == 'SemanticPointer':
        sim = np.tensordot(sp.v, vocab_vectors, axes=([0], [1]))
    else:
        sim = np.tensordot(sp, vocab_vectors, axes=([0], [1]))

    sim_true = np.tensordot(item, vocab_vectors, axes=([0], [1]))

    ind = np.argmax(sim)

    ind_true = np.argmax(sim_true)

    if sim[ind] < sim_threshold:
        if ind == ind_true:
            print("Warning: closest match is correct, but returning 0 due to it being below threshold")
        return 0

    if ind == ind_true:
        return 1
    else:
        return 0


def loc_match(sp, heatmap_vectors, coord, xs, ys, distance_threshold=0.5, sim_threshold=0.0):
    if sp.__class__.__name__ == 'SemanticPointer':
        vs = np.tensordot(sp.v, heatmap_vectors, axes=([0], [2]))
    else:
        vs = np.tensordot(sp, heatmap_vectors, axes=([0], [2]))

    xy = np.unravel_index(vs.argmax(), vs.shape)

    x = xs[xy[0]]
    y = ys[xy[1]]

    # Not similar enough to anything, so count as incorrect
    if vs[xy] < sim_threshold:
        return 0

    # If within threshold of the correct location, count as correct
    if (x-coord[0])**2 + (y-coord[1])**2 < distance_threshold**2:
        return 1
    else:
        return 0


def loc_match_duplicate(sp, heatmap_vectors, coord1, coord2, xs, ys,
                        distance_threshold=0.5, sim_threshold=0.0, zero_range=8,
                        ):
    """
    Checks that two locations are represented in an SSP
    :param sp: the spatial semantic pointer
    :param heatmap_vectors:
    :param coord1: the true coordinate of the first item
    :param coord2: the true coordinate of the second item
    :param xs: linspace in x
    :param ys: linspace in y
    :param distance_threshold: if the predicted location is within this threshold of the true location it is correct
    :param sim_threshold: if the similarity is below this value it is incorrect, regardless of location
    :param zero_range: size of the square of indices around the first peak that gets zeroed out.
    :return:
    """

    if sp.__class__.__name__ == 'SemanticPointer':
        vs = np.tensordot(sp.v, heatmap_vectors, axes=([0], [2]))
    else:
        vs = np.tensordot(sp, heatmap_vectors, axes=([0], [2]))

    xy = np.unravel_index(vs.argmax(), vs.shape)

    # Not similar enough to anything, so count as incorrect
    if vs[xy] < sim_threshold:
        return 0

    score = 0.

    x = xs[xy[0]]
    y = ys[xy[1]]

    # Check if it found the first coordinate
    if (x - coord1[0]) ** 2 + (y - coord1[1]) ** 2 < distance_threshold ** 2:
        # Check if both points at the same location, if so, count them both as correct
        if (coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 < distance_threshold ** 2:
            return 1

        score += 0.5

        # Explicitly zero-out around the peak
        x1 = max(0, xy[0] - zero_range)
        x2 = min(len(xs), xy[0] + zero_range + 1)
        y1 = max(0, xy[1] - zero_range)
        y2 = min(len(ys), xy[1] + zero_range + 1)
        vs[x1:x2, y1:y2] = 0

        xy = np.unravel_index(vs.argmax(), vs.shape)

        if vs[xy] < sim_threshold:
            return score

        x = xs[xy[0]]
        y = ys[xy[1]]

        if (x - coord2[0]) ** 2 + (y - coord2[1]) ** 2 < distance_threshold ** 2:
            return 1

    elif (x - coord2[0]) ** 2 + (y - coord2[1]) ** 2 < distance_threshold ** 2:
        # Check if both points at the same location, if so, count them both as correct
        if (coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 < distance_threshold ** 2:
            return 1

        score += 0.5

        # Explicitly zero-out around the peak
        x1 = max(0, xy[0] - zero_range)
        x2 = min(len(xs), xy[0] + zero_range + 1)
        y1 = max(0, xy[1] - zero_range)
        y2 = min(len(ys), xy[1] + zero_range + 1)
        vs[x1:x2, y1:y2] = 0

        xy = np.unravel_index(vs.argmax(), vs.shape)

        if vs[xy] < sim_threshold:
            return score

        x = xs[xy[0]]
        y = ys[xy[1]]

        if (x - coord1[0]) ** 2 + (y - coord1[1]) ** 2 < distance_threshold ** 2:
            return 1

    return score


def region_item_match(sp, vocab_vectors, vocab_indices, sim_threshold=0.5):
    if sp.__class__.__name__ == 'SemanticPointer':
        sim = np.tensordot(sp.v, vocab_vectors, axes=([0], [1]))
    else:
        sim = np.tensordot(sp, vocab_vectors, axes=([0], [1]))

    n_matches = len(vocab_indices)

    # sorts from lowest to highest by default
    indices = np.argsort(sim)
    # reverse to have highest to lowest
    indices = indices[::-1]

    # If nothing should be inside the region
    if n_matches == 0:
        if sim[indices[0]] < sim_threshold:
            return 1
        else:
            return 0

    acc = 0

    for i, ind in enumerate(indices):
        if i < n_matches:
            # Should be in the region and detected in region
            if ind in vocab_indices:
                acc += 1
        else:
            # Should be outside the region and detected outside region
            if ind not in vocab_indices:
                acc += 1

    acc /= vocab_vectors.shape[0]

    return acc
