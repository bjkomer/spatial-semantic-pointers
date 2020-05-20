import numpy as np
import nengo
import nengo_spa as spa
import struct
from functools import reduce


def power(s, e):
    x = np.fft.ifft(np.fft.fft(s.v) ** e).real
    return spa.SemanticPointer(data=x)


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
    return spa.SemanticPointer(v)


def get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp):
    """
    Precompute spatial semantic pointers for every location in the linspace
    Used to quickly compute heat maps by a simple vectorized dot product (matrix multiplication)
    """
    if x_axis_sp.__class__.__name__ == 'SemanticPointer':
        dim = len(x_axis_sp.v)
    else:
        dim = len(x_axis_sp)
        x_axis_sp = spa.SemanticPointer(data=x_axis_sp)
        y_axis_sp = spa.SemanticPointer(data=y_axis_sp)

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
    n_samples = sps.shape[0]

    # Compute the dot product of every semantic pointer with every element in the heatmap
    # vs will be of shape (n_samples, res_x, res_y)
    vs = np.tensordot(sps, heatmap_vectors, axes=([-1], [2]))

    # Find the x and y indices for every sample. xys is a list of two elements.
    # Each element in a numpy array of shape (n_samples,)
    xys = np.unravel_index(vs.reshape((n_samples, res_x * res_y)).argmax(axis=1), (res_x, res_y))

    # Transform into an array containing coordinates
    # locs will be of shape (n_samples, 2)
    locs = np.vstack([xs[xys[0]], ys[xys[1]]]).T

    assert(locs.shape[0] == n_samples)
    assert(locs.shape[1] == 2)

    return locs


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

    sp = spa.SemanticPointer(data=vector)

    if normalize:
        try:
            sp = sp.normalized()
        except:
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
        return spa.SemanticPointer(data=vec)
    else:
        return vec

#################
# Periodic Axes #
#################


# modified from 'make_good_unitary' from arvoelke
def make_fixed_dim_periodic_axis(dim=128, period=4, phase=0, frequency=1,
                                 eps=1e-3, rng=np.random, flip=False,
                                 random_phases=False,
                                 ):
    # will repeat at a distance of 2*spacing
    # dimensionality is fixed

    # phi_list = np.linspace(0, np.pi, spacing + 1)[1:-1]

    val = 2 * np.pi / period
    phi_list = [val, -val]

    i = 2
    while i * val <= np.pi:
        phi_list.append(i * val)
        phi_list.append(-i * val)
        i += 1

    if random_phases:
        phase = rng.uniform(-np.pi, np.pi, size=(dim - 1) // 2)

    phi = rng.choice(phi_list, replace=True, size=(dim - 1) // 2)

    # assert np.all(np.abs(phi) >= np.pi * eps)
    # assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    if flip:
        fv[1:(dim + 1) // 2] = np.sin(phi * frequency + phase) + 1j * np.cos(phi * frequency + phase)
    else:
        fv[1:(dim + 1) // 2] = np.cos(phi * frequency + phase) + 1j * np.sin(phi * frequency + phase)
    fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])
    if dim % 2 == 0:
        fv[dim // 2] = 1  #hmmm... does this have any implications?

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    # assert np.allclose(v.imag, 0, atol=1e-5)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v
    # return spa.SemanticPointer(v)


def make_optimal_periodic_axis(dim=128, eps=1e-3, phase=0, rng=np.random):
    spacing = dim // 2

    phi = np.linspace(0, np.pi, spacing + 1)[1:-1]

    # Randomize the order of the phi, so that different vectors can be created
    rng.shuffle(phi)

    assert np.all(np.abs(phi) >= np.pi * eps)
    assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    fv[1:(dim + 1) // 2] = np.cos(phi + phase) + 1j * np.sin(phi + phase)
    fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])
    if dim % 2 == 0:
        fv[dim // 2] = 1

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return spa.SemanticPointer(v)


##################################
# Hexagonal Coordinate Functions #
##################################


def rotate_vector_along_axis(vec, rot_axis, theta):
    axis = rot_axis.copy()
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.dot(np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                            [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                            [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]]),
                  vec)


grid_angle = 0

normal = np.array([1, 1, 1])
default_x_axis = np.array([1, -1, 0])
default_y_axis = np.array([-1, -1, 2])
normal = normal / np.linalg.norm(normal)
default_x_axis = default_x_axis / np.linalg.norm(default_x_axis)
default_y_axis = default_y_axis / np.linalg.norm(default_y_axis)

default_x_axis = rotate_vector_along_axis(default_x_axis, normal, grid_angle * np.pi / 180)
default_y_axis = rotate_vector_along_axis(default_y_axis, normal, grid_angle * np.pi / 180)

# Used in the vectorized functions
default_xy_axes = np.vstack([default_x_axis, default_y_axis]).T


def xyz_to_xy(coord, x_axis=default_x_axis, y_axis=default_y_axis):
    """
    Projects a 3D hexagonal coordinate into the
    corresponding 2D coordinate
    """
    x = np.dot(x_axis, coord)
    y = np.dot(y_axis, coord)

    return np.array([x, y])


def xyz_to_xy_v(coords, xy_axes=default_xy_axes):
    """
    Projects a 3D hexagonal coordinate into the
    corresponding 2D coordinate

    coords is a (n, 3) matrix
    xy_axes is a (3, 2) matrix
    """

    return np.dot(coords, xy_axes)


def xy_to_xyz(coord, x_axis=default_x_axis, y_axis=default_y_axis):
    """
    Converts a 2D coordinate into the corresponding
    3D coordinate in the hexagonal representation
    """
    return x_axis*coord[0]+y_axis*coord[1]


def xy_to_xyz_v(coords, xy_axes=default_xy_axes):
    """
    Converts a 2D coordinate into the corresponding
    3D coordinate in the hexagonal representation
    coord is a (n, 2) matrix
    xy_axes is a (3, 2) matrix
    """

    return np.dot(coords, xy_axes.T)


def encode_point_hex(x, y, x_axis_sp, y_axis_sp, z_axis_sp):
    """
    Encodes a given 2D point as a 3D hexagonal SSP
    """
    xyz = xy_to_xyz((x, y))

    return power(x_axis_sp, xyz[0]) * power(y_axis_sp, xyz[1]) * power(z_axis_sp, xyz[2])


def get_heatmap_vectors_hex(xs, ys, x_axis_sp, y_axis_sp, z_axis_sp):
    """
    Precompute spatial semantic pointers for every location in the linspace
    Used to quickly compute heat maps by a simple vectorized dot product (matrix multiplication)
    """
    if x_axis_sp.__class__.__name__ == 'SemanticPointer':
        dim = len(x_axis_sp.v)
    else:
        dim = len(x_axis_sp)
        x_axis_sp = spa.SemanticPointer(data=x_axis_sp)
        y_axis_sp = spa.SemanticPointer(data=y_axis_sp)
        z_axis_sp = spa.SemanticPointer(data=z_axis_sp)

    vectors = np.zeros((len(xs), len(ys), dim))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = encode_point_hex(
                x=x, y=y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp, z_axis_sp=z_axis_sp
            )
            vectors[i, j, :] = p.v

    return vectors

#################################################
# N-dimensional projection coordinate functions #
#################################################


# def encode_point_n(x, y, axis_sps):
#     """
#     Encodes a given 2D point as a ND SSP
#     """
#
#     n = len(axis_sps)
#
#     x_axis = np.ones((n,))
#     y_axis = np.ones((n,))
#
#     # There are many ways to create orthogonal vectors, this is just one of them
#     if n % 2 == 0:
#         # every other element negative, with the last two elements as 0
#         x_axis[1::2] = -1
#         x_axis[-2:] = 0
#         # all but the last two elements -1, last element equal to n-2
#         y_axis[:-2] = -1
#         y_axis[-2] = 0
#         y_axis[-1] = n - 2
#     else:
#         # every other element negative, with the last element as 0
#         x_axis[1::2] = -1
#         x_axis[-1] = 0
#         # all but the last element -1, last element equal to n-1
#         y_axis[:-1] = -1
#         y_axis[-1] = n - 1
#
#     # doublecheck that everything worked as expected
#     assert (np.dot(x_axis, y_axis) == 0)
#     assert (np.dot(x_axis, np.ones((n,))) == 0)
#     assert (np.dot(y_axis, np.ones((n,))) == 0)
#
#     x_axis = x_axis / np.linalg.norm(x_axis)
#     y_axis = y_axis / np.linalg.norm(y_axis)
#
#     # 2D point represented as an N dimensional vector in the plane spanned by 'x_axis' and 'y_axis'
#     vec = x_axis * x + y_axis * y
#
#     # Generate the SSP from the high dimensional vector, by convolving all of the axis vector components together
#     ret = power(axis_sps[0], vec[0])
#     for i in range(1, n):
#         ret *= power(axis_sps[i], vec[i])
#
#     return ret

def encode_point_n(x, y, axis_sps):
    """
    Encodes a given 2D point as a ND SSP
    """

    N = len(axis_sps)

    points_nd = np.zeros((N + 1, N))
    points_nd[:N, :] = np.eye(N)
    # points in 2D that will correspond to each axis, plus one at zero
    points_2d = np.zeros((N + 1, 2))
    thetas = np.linspace(0, 2 * np.pi, N + 1)[:-1]
    # TODO: will want a scaling here, or along the high dim axes
    for i, theta in enumerate(thetas):
        points_2d[i, 0] = np.cos(theta)
        points_2d[i, 1] = np.sin(theta)

    transform_mat = np.linalg.lstsq(points_2d, points_nd)

    # apply scaling to the axes based on the singular values. Both should be the same
    x_axis = transform_mat[0][0, :] / transform_mat[3][0]
    y_axis = transform_mat[0][1, :] / transform_mat[3][1]

    # 2D point represented as an N dimensional vector in the plane spanned by 'x_axis' and 'y_axis'
    # apply appropriate scaling based on the singular value
    vec = (x_axis * x + y_axis * y) * transform_mat[3][0]

    # Generate the SSP from the high dimensional vector, by convolving all of the axis vector components together
    ret = power(axis_sps[0], vec[0])
    for i in range(1, N):
        ret *= power(axis_sps[i], vec[i])

    return ret


def get_heatmap_vectors_n(xs, ys, n, seed=13, dim=512):
    """
    Precompute spatial semantic pointers for every location in the linspace
    Used to quickly compute heat maps by a simple vectorized dot product (matrix multiplication)
    """
    rng = np.random.RandomState(seed=seed)
    axis_sps = []
    for i in range(n):
        axis_sps.append(make_good_unitary(dim, rng=rng))

    vectors = np.zeros((len(xs), len(ys), dim))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = encode_point_n(
                x=x, y=y, axis_sps=axis_sps
            )
            vectors[i, j, :] = p.v

    # also return the axis_sps so individual points can be generated
    return vectors, axis_sps


def get_axes(dim=256, n=3, seed=13, period=0, optimal_phi=False):
    """
    Get X and Y axis vectors based on an n dimensional projection.
    If spacing is non-zero, they will be periodic with the given spacing
    :param dim:
    :param n:
    :param seed:
    :param spacing:
    :return:
    """
    rng = np.random.RandomState(seed=seed)

    # # Length of the normal vector to the plane
    # len_normal = np.linalg.norm(np.ones((n,)) * 1./n)
    # # pythagorean theorem to find the length along the axis, assuming in the 2D space the length to the point is one
    # len_axis = np.sqrt(len_normal**2 + 1)

    points_nd = np.eye(n) #* np.sqrt(n)
    # points in 2D that will correspond to each axis, plus one at zero
    points_2d = np.zeros((n, 2))
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    # TODO: will want a scaling here, or along the high dim axes
    for i, theta in enumerate(thetas):
        points_2d[i, 0] = np.cos(theta)
        points_2d[i, 1] = np.sin(theta)

    transform_mat = np.linalg.lstsq(points_2d, points_nd)

    x_axis = transform_mat[0][0, :]
    y_axis = transform_mat[0][1, :]

    axis_sps = []
    for i in range(n):
        if period == 0:
            if optimal_phi:
                # unitary vector with unique phi for every element
                # periodicity depends on the dimensionality
                axis_sps.append(
                    make_optimal_periodic_axis(dim, rng=rng)
                )
            else:
                # random unitary vector
                axis_sps.append(
                    make_good_unitary(dim, rng=rng)
                )
        else:
            # unitary vector of fixed dimensionality and fixed periodicity
            axis_sps.append(
                spa.SemanticPointer(data=make_fixed_dim_periodic_axis(dim=dim, period=period, rng=rng))
            )

    X = power(axis_sps[0], x_axis[0])
    Y = power(axis_sps[0], y_axis[0])
    for i in range(1, n):
        X *= power(axis_sps[i], x_axis[i])
        Y *= power(axis_sps[i], y_axis[i])

    return X, Y


def get_all_factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def get_fixed_dim_grid_axes(dim=256, seed=13, scale_min=1, scale_max=2):
    rng = np.random.RandomState(seed=seed)

    if dim % 2 == 0:
        wdim = dim - 2
    else:
        wdim = dim - 1

    assert wdim % 6 == 0

    # # attempt to define rotations and scales as evenly as possible
    # # erring on the side of more scales than rotations
    #
    # factors = np.array(get_all_factors(wdim // 6))
    #
    # # an ideal factor will be close to the square root of the number
    # ideal = int(np.sqrt(wdim // 6))
    #
    # # find the index of this closest factor
    # ind = np.argmin(np.abs(factors - ideal))
    #
    # # construct a list with this factor and 'conjugate'
    # div = [factors[ind], int((wdim // 6) / factors[ind])]
    #
    # # the number of scales will be the bigger one
    # n_scales = max(div)
    #
    # # the number of rotations per scale will be the smaller one
    # n_rotations = min(div)

    # using the maximum possible number of scales
    n_scales = wdim // 6
    n_rotations = 1

    scales = rng.uniform(scale_min, scale_max, (n_scales,))

    # 3 directions 120 degrees apart
    vec_dirs = [0, 2 * np.pi / 3, 4 * np.pi / 3]

    # random rotations of the triplets of vectors that sum to 0
    rotations = rng.uniform(-np.pi, np.pi, (n_scales, n_rotations))

    fxy = np.zeros((2, dim), dtype='complex64')

    # first element must be 1 or -1
    fxy[:, 0] = 1

    # if even dim, middle element must be 1 or -1
    if dim % 2 == 0:
        fxy[:, dim // 2] = 1

    for i in range(2):
        for s in range(n_scales):
            for r in range(n_rotations):
                kvecs = np.array([
                    [np.sin(vec_dirs[0] + rotations[s, r]), np.cos(vec_dirs[0] + rotations[s, r])],
                    [np.sin(vec_dirs[1] + rotations[s, r]), np.cos(vec_dirs[1] + rotations[s, r])],
                    [np.sin(vec_dirs[2] + rotations[s, r]), np.cos(vec_dirs[2] + rotations[s, r])],
                ])
                for k in range(len(kvecs)):
                    fxy[i, 1 + s * n_rotations * 3 + r * 3 + k] = np.exp(1.j * (scales[s] * kvecs[k, i]))

        # set all of the complex conjugates
        fxy[i, -1:dim // 2:-1] = np.conj(fxy[i, 1:(dim + 1) // 2])

    assert np.allclose(np.abs(fxy[0, :]), 1)
    assert np.allclose(np.abs(fxy[1, :]), 1)

    x = np.fft.ifft(fxy[0, :]).real
    y = np.fft.ifft(fxy[1, :]).real

    return spa.SemanticPointer(x), spa.SemanticPointer(y)


def get_proj_phi(n_proj, phi, angle):
    # get the phis corresponding to an orthogonal N-D axis projection onto a 2-D space
    # TODO: this function can be made much more efficient

    # minimal dimensionality required for this representation
    dim = n_proj * 2 + 1

    # make an axis for each n_proj
    axes_f = np.zeros((n_proj, dim), dtype='complex64')
    axes = np.zeros((n_proj, dim,))
    axis_sps = []
    for k in range(n_proj):
        axes_f[k, :] = 1
        axes_f[k, k + 1] = np.exp(1.j * phi)
        axes_f[k, -(k + 1)] = np.conj(axes_f[k, k + 1])
        axes[k, :] = np.fft.ifft(axes_f[k, :]).real

        assert np.allclose(np.abs(axes_f[k, :]), 1)
        assert np.allclose(np.fft.fft(axes[k, :]), axes_f[k, :])
        assert np.allclose(np.linalg.norm(axes[k, :]), 1)

        axis_sps.append(spa.SemanticPointer(data=axes[k, :]))

    points_nd = np.zeros((n_proj + 1, n_proj))
    points_nd[:n_proj, :] = np.eye(n_proj) * np.sqrt(n_proj)
    # points in 2D that will correspond to each axis, plus one at zero
    # the zero isn't really necessary, but doesn't hurt
    points_2d = np.zeros((n_proj + 1, 2))
    if n_proj != 2:
        thetas = np.linspace(0, 2 * np.pi, n_proj + 1)[:-1] + angle
    else:  # special case for 2D, don't want polar opposite axes
        thetas = np.linspace(0, np.pi, n_proj + 1)[:-1] + angle

    for i, theta in enumerate(thetas):
        points_2d[i, 0] = np.cos(theta)
        points_2d[i, 1] = np.sin(theta)

    transform_mat = np.linalg.lstsq(points_2d, points_nd)

    # apply scaling to the axes based on the singular values. Both should be the same
    # note: this scaling seems only relevant to matching the effective random scaling, and needed on both sides anyway
    x_axis = transform_mat[0][0, :] #/ transform_mat[3][0]
    y_axis = transform_mat[0][1, :] #/ transform_mat[3][1]

    X = power(axis_sps[0], x_axis[0])
    Y = power(axis_sps[0], y_axis[0])
    for i in range(1, n_proj):
        X *= power(axis_sps[i], x_axis[i])
        Y *= power(axis_sps[i], y_axis[i])

    xf = np.fft.fft(X.v)
    yf = np.fft.fft(Y.v)

    return xf[1:1+n_proj], yf[1:1+n_proj]


def get_fixed_dim_sub_toriod_axes(
        dim=256,
        n_proj=3,
        scale_ratio=(1 + 5 ** 0.5) / 2,
        scale_start_index=0,
        rng=np.random,
        eps=0.001,
):

    # number of toroids possible given the projection dimension
    # if this does not evenly divide dim, extra dimensions will be set to zero (1 in frequency domain)
    n_toroids = ((dim-1)//2)//n_proj

    # Randomly select the angle for each toroid
    angles = rng.uniform(0, 2*np.pi, size=(n_toroids,))

    # if the scale ration is set to 0, use a random distribution instead
    if scale_ratio == 0:
        scales = rng.uniform(-np.pi + eps, np.pi - eps, size=n_toroids)
    else:
        # create scales starting from the largest scale and moving down based on the ratio given with a start index of 0
        # the first scale will always be 2pi (minus epsilon to make the direction non-ambiguous)
        # this is effectively the finest resolution that can be detected (though the scale into this rep is arbitrary)
        scales = np.array([(np.pi - eps)/(scale_ratio**i) for i in range(scale_start_index, n_toroids+scale_start_index)])

    xf = np.ones((dim,), dtype='complex64')
    yf = np.ones((dim,), dtype='complex64')

    for n in range(n_toroids):
        phis_x, phis_y = get_proj_phi(n_proj=n_proj, phi=scales[n], angle=angles[n])
        xf[1 + n * n_proj:1 + (n + 1) * n_proj] = phis_x
        yf[1 + n * n_proj:1 + (n + 1) * n_proj] = phis_y

    # set the appropriate conjugates
    xf[-1:dim // 2:-1] = np.conj(xf[1:(dim + 1) // 2])
    yf[-1:dim // 2:-1] = np.conj(yf[1:(dim + 1) // 2])
    if dim % 2 == 0:
        xf[dim // 2] = 1
        yf[dim // 2] = 1

    assert np.allclose(np.abs(xf), 1)
    assert np.allclose(np.abs(yf), 1)
    x = np.fft.ifft(xf).real
    y = np.fft.ifft(yf).real
    assert np.allclose(np.fft.fft(x), xf)
    assert np.allclose(np.fft.fft(y), yf)
    assert np.allclose(np.linalg.norm(x), 1)
    assert np.allclose(np.linalg.norm(y), 1)

    return spa.SemanticPointer(x), spa.SemanticPointer(y)


def get_fixed_dim_variable_sub_toriod_axes(
        dim=256,
        rng=np.random,
        eps=0.001,
):

    # total number of components that can be used
    n_circles = (dim-1)//2

    # Generate potiential sub-manifold dimensions.
    # Commonly selecting 3, but allowing higher dimensions
    # toroid_dims = np.ceil(1 + 2 * np.random.poisson(1, size=n_circles))
    toroid_dims = np.floor(3 + np.random.poisson(1, size=n_circles))

    # trim any excess, make the last dimension fit the remaining space
    cumulative = 0
    for i in range(n_circles):
        if cumulative + toroid_dims[i] >= n_circles:
            toroid_dims[i] = n_circles - cumulative
            toroid_dims = toroid_dims[:i+1].copy()
            n_toroids = i + 1
            break
        else:
            cumulative += toroid_dims[i]

    # Randomly select the angle for each toroid
    angles = rng.uniform(0, 2*np.pi, size=(n_toroids,))

    scales = rng.uniform(-np.pi + eps, np.pi - eps, size=n_toroids)

    xf = np.ones((dim,), dtype='complex64')
    yf = np.ones((dim,), dtype='complex64')

    for n in range(n_toroids):
        n_proj = int(toroid_dims[n])
        phis_x, phis_y = get_proj_phi(n_proj=n_proj, phi=scales[n], angle=angles[n])
        xf[1 + n * n_proj:1 + (n + 1) * n_proj] = phis_x
        yf[1 + n * n_proj:1 + (n + 1) * n_proj] = phis_y

    # set the appropriate conjugates
    xf[-1:dim // 2:-1] = np.conj(xf[1:(dim + 1) // 2])
    yf[-1:dim // 2:-1] = np.conj(yf[1:(dim + 1) // 2])
    if dim % 2 == 0:
        xf[dim // 2] = 1
        yf[dim // 2] = 1

    assert np.allclose(np.abs(xf), 1)
    assert np.allclose(np.abs(yf), 1)
    x = np.fft.ifft(xf).real
    y = np.fft.ifft(yf).real
    assert np.allclose(np.fft.fft(x), xf)
    assert np.allclose(np.fft.fft(y), yf)
    assert np.allclose(np.linalg.norm(x), 1)
    assert np.allclose(np.linalg.norm(y), 1)

    return spa.SemanticPointer(x), spa.SemanticPointer(y)
