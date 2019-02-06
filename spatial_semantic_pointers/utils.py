import numpy as np
import nengo


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


def generate_region_vector(desired, xs, ys, x_axis_sp, y_axis_sp):
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
    sp.normalize()

    return sp
