import numpy as np
import nengo
import struct


def power(s, e):
    x = np.fft.ifft(np.fft.fft(s.v) ** e).real
    return nengo.spa.SemanticPointer(data=x)


def encode_point(x, y, x_axis_sp, y_axis_sp):

    return power(x_axis_sp, x) * power(y_axis_sp, y)


def spatial_dot(vec, xs, ys, x_axis_sp, y_axis_sp):
    vs = np.zeros((len(ys), len(xs)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = encode_point(
                x=x, y=y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
            )
            # Allow support for both vectors and semantic pointers as input
            if vec.__class__.__name__ == 'SemanticPointer':
                vs[j, i] = np.dot(vec.v, p.v)
            else:
                vs[j, i] = np.dot(vec, p.v)
    return vs


def make_good_unitary(dim, eps=1e-3, rng=np.random):
    # created by arvoelke
    a = rng.rand((dim - 1) // 2)
    sign = rng.choice((-1, +1), len(a))
    phi = sign * np.pi * (eps + a * (1 - 2 * eps))
    assert np.all(np.abs(phi) >= np.pi * eps)
    assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    fv[1:(dim + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
    fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])
    if dim % 2 == 0:
        fv[dim // 2] = 1

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    # assert np.allclose(v.imag, 0, atol=1e-5)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return nengo.spa.SemanticPointer(v)


def get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp):
    """
    Precompute spatial semantic pointers for every location in the linspace
    Used to quickly compute heat maps by a simple vectorized dot product (matrix multiplication)
    """
    if x_axis_sp.__class__.__name__ == 'SemanticPointer':
        dim = len(x_axis_sp.v)
    else:
        dim = len(x_axis_sp)
        x_axis_sp = nengo.spa.SemanticPointer(data=x_axis_sp)
        y_axis_sp = nengo.spa.SemanticPointer(data=y_axis_sp)

    vectors = np.zeros((len(xs), len(ys), dim))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = encode_point(
                x=x, y=y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
            )
            vectors[i, j, :] = p.v

    return vectors


def rotate_vector(start_vec, end_vec, theta):
    """
    Rotate a vector starting at 'start_vec' in the plane formed by 'start_vec' and 'end_vec'
    in a direction toward 'end_vec' with an angle of 'theta'
    Returns a new vector that is the result of the rotation
    """
    A_prime = start_vec / np.linalg.norm(start_vec)

    B_tilde = end_vec - np.dot(A_prime, end_vec) * A_prime

    # Orthogonal normalized vector
    B_prime = B_tilde / np.linalg.norm(B_tilde)

    C = np.linalg.norm(start_vec) * ((np.cos(theta) * A_prime + np.sin(theta) * B_prime))

    C_prime = C / np.linalg.norm(C)

    return C_prime


def ssp_to_loc(sp, heatmap_vectors, xs, ys):
    """
    Convert an SSP to the approximate location that it represents.
    Uses the heatmap vectors as a lookup table
    :param sp: the semantic pointer of interest
    :param heatmap_vectors: SSP for every point in the space defined by xs and ys
    :param xs: linspace in x
    :param ys: linspace in y
    :return: 2D coordinate that the SSP most closely represents
    """

    if sp.__class__.__name__ == 'SemanticPointer':
        vs = np.tensordot(sp.v, heatmap_vectors, axes=([0], [2]))
    else:
        vs = np.tensordot(sp, heatmap_vectors, axes=([0], [2]))

    xy = np.unravel_index(vs.argmax(), vs.shape)

    x = xs[xy[0]]
    y = ys[xy[1]]

    return np.array([x, y])


def ssp_to_loc_v(sps, heatmap_vectors, xs, ys):
    """
    vectorized version of ssp_to_loc
    Convert an SSP to the approximate location that it represents.
    Uses the heatmap vectors as a lookup table
    :param sps: array of spatial semantic pointers of interest
    :param heatmap_vectors: SSP for every point in the space defined by xs and ys
    :param xs: linspace in x
    :param ys: linspace in y
    :return: array of the 2D coordinates that the SSP most closely represents
    """

    assert(len(sps.shape) == 2)
    assert(len(heatmap_vectors.shape) == 3)
    assert(sps.shape[1] == heatmap_vectors.shape[2])

    res_x = heatmap_vectors.shape[0]
    res_y = heatmap_vectors.shape[1]
    dim = heatmap_vectors.shape[2]
    n_samples = sps.shape[0]

    # Compute the dot product of every semantic pointer with every element in the heatmap
    # vs will be of shape (n_samples, res_x, res_y)
    vs = np.tensordot(sps, heatmap_vectors, axes=([-1], [2]))

    # Find the x and y indices for every sample. xys is a list of two elements.
    # Each element in a numpy array of shape (n_samples,)
    xys = np.unravel_index(vs.reshape((dim, res_x*res_y)).argmax(axis=1), (res_x, res_y))

    # Transform into an array containing coordinates
    # locs will be of shape (n_samples, 2)
    locs = np.vstack([xs[xys[0]], ys[xys[1]]]).T

    assert(locs.shape[0] == n_samples)
    assert(locs.shape[1] == 2)

    return locs

    # x = xs[xy[0]]
    # y = ys[xy[1]]
    #
    # return np.array([x, y])


####################
# Region Functions #
####################

def generate_region_vector(desired, xs, ys, x_axis_sp, y_axis_sp, normalize=True):
    """
    :param desired: occupancy grid of what points should be in the region and which ones should not be
    :param xs: linspace in x
    :param ys: linspace in y
    :param x_axis_sp: x axis semantic pointer
    :param y_axis_sp: y axis semantic pointer
    :return: a normalized semantic pointer designed to be highly similar to the desired region
    """

    vector = np.zeros_like((x_axis_sp.v))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if desired[i, j] == 1:
                vector += encode_point(x, y, x_axis_sp, y_axis_sp).v

    sp = nengo.spa.SemanticPointer(data=vector)

    if normalize:
        sp.normalize()

    return sp


def circular_region(xs, ys, radius=1, x_offset=0, y_offset=0):
    region = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if (x - x_offset)**2 + (y - y_offset)**2 < radius**2:
                region[j, i] = 1

    return region


def generate_circular_region_vector(xs, ys, x_axis_sp, y_axis_sp, radius=1, x_offset=0, y_offset=0, normalize=True):
    return generate_region_vector(
        desired=circular_region(xs, ys, radius=radius, x_offset=x_offset, y_offset=y_offset),
        xs=xs, ys=ys, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp, normalize=normalize,
    )


def elliptic_region(xs, ys, f1, f2, diameter=1):
    """
    :param xs: linspace in x
    :param ys: linspace in y
    :param f1: first focal point (x,y)
    :param f2: second focal point (x,y)
    :param diameter: length of the major axis
    :return: occupancy grid defining a filled-in ellipse for the given space
    """
    region = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if np.sqrt((x - f1[0])**2 + (y - f1[1])**2) + np.sqrt((x - f2[0])**2 + (y - f2[1])**2) < diameter:
                region[j, i] = 1
    return region


def generate_elliptic_region_vector(xs, ys, x_axis_sp, y_axis_sp, f1, f2, diameter=1, normalize=True):
    return generate_region_vector(
        desired=elliptic_region(xs, ys, f1=f1, f2=f2, diameter=diameter),
        xs=xs, ys=ys, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp, normalize=normalize,
    )


def simplify_angle(theta):
    """
    Convert the given angle to be between -pi and pi
    """
    while theta > np.pi:
        theta -= 2*np.pi
    while theta < -np.pi:
        theta += 2*np.pi
    return theta


def arc_region(xs, ys, arc_center, arc_width, x_offset=0, y_offset=0):
    region = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            angle = np.arctan2(x - x_offset, y - y_offset)
            diff = simplify_angle(angle - arc_center)
            if diff < arc_width / 2. and diff > -arc_width / 2.:
                region[j, i] = 1

    return region


def generate_arc_region_vector(xs, ys, x_axis_sp, y_axis_sp, arc_center, arc_width, x_offset=0, y_offset=0, normalize=True):
    return generate_region_vector(
        desired=arc_region(xs, ys, arc_center=arc_center, arc_width=arc_width, x_offset=x_offset, y_offset=y_offset),
        xs=xs, ys=ys, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp, normalize=normalize,
    )


def encode_random(x, y, dim=512, convert_to_sp=False):
    """
    Used for comparison with SSPs. A deterministic random encoding of a location to a semantic pointer
    """
    # convert x and y into a single float
    f = x * 1000 + y
    # convert the float into an unsigned integer to be used as a seed
    seed = struct.unpack('>I', struct.pack('>f', f))[0]
    rstate = np.random.RandomState(seed)
    vec = rstate.normal(size=dim)
    vec = vec / np.linalg.norm(vec)
    if convert_to_sp:
        return nengo.spa.SemanticPointer(data=vec)
    else:
        return vec
