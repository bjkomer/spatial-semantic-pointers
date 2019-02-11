import numpy as np
import sys
from PIL import Image
import base64
import matplotlib.cm as cm
import seaborn as sns

# Python version specific imports
if sys.version_info[0] == 3:
    from io import BytesIO
else:
    import cStringIO


def image_svg(arr):
    """
    Given an array, return an svg image
    """
    if sys.version_info[0] == 3:
        # Python 3

        png = Image.fromarray(arr)
        buffer = BytesIO()
        png.save(buffer, format="PNG")
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        return '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))

    else:
        # Python 2

        png = Image.fromarray(arr)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        return '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))


class SpatialHeatmap(object):

    def __init__(self, heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None):

        self.heatmap_vectors = heatmap_vectors
        self.xs = xs
        self.ys = ys

        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

        self.cm = cm.get_cmap(cmap)

        self._nengo_html_ = ""

    def __call__(self, t, x):

        if len(self.heatmap_vectors.shape) == 4:
            # Index for selecting which heatmap vectors to use
            # Used for deconvolving with multiple items in the same plot
            index = int(x[0])

            hmv = self.heatmap_vectors[index, ...]
            vector = x[1:]
        else:
            hmv = self.heatmap_vectors
            vector = x

        # Generate heatmap values
        vs = np.tensordot(vector, hmv, axes=([0], [2]))

        if self.vmin is None:
            min_val = np.min(vs)
        else:
            min_val = self.vmin

        if self.vmax is None:
            max_val = np.max(vs)
        else:
            max_val = self.vmax

        vs = np.clip(vs, a_min=min_val, a_max=max_val)

        xy = np.unravel_index(vs.argmax(), vs.shape)

        values = (self.cm(vs)*255).astype(np.uint8)

        self._nengo_html_ = image_svg(values)


def plot_predictions(predictions, coords, ax, min_val=-1, max_val=1):
    """
    plot predictions, and colour them based on their true coords
    both predictions and coords are (n_samples, 2) vectors
    """

    for n in range(predictions.shape[0]):
        x = predictions[n, 0]
        y = predictions[n, 1]

        # Note: this clipping shouldn't be necessary
        xa = np.clip(coords[n, 0], min_val, max_val)
        ya = np.clip(coords[n, 1], min_val, max_val)

        r = float(((xa - min_val) / (max_val - min_val)))
        # g = float(((ya - min_val) / (max_val - min_val)))
        b = float(((ya - min_val) / (max_val - min_val)))

        # ax.scatter(x, y, color=(r, g, 0))
        ax.scatter(x, y, color=(r, 0, b))

    return ax


def plot_heatmap(vec, heatmap_vectors, ax, xs, ys, name='', vmin=-1, vmax=1, cmap='plasma', invert=False):
    # vs = np.dot(vec, heatmap_vectors)
    # vec has shape (dim) and heatmap_vectors have shape (xs, ys, dim) so the result will be (xs, ys)
    vs = np.tensordot(vec, heatmap_vectors, axes=([0], [2]))

    if cmap == 'diverging':
        cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)

    if invert:
        img = ax.imshow(vs, interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1]), vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        img = ax.imshow(vs, origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1]), vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_title(name)

    return img


def plot_vocab_similarity(vec, vocab_vectors, ax):

    sim = np.tensordot(vec, vocab_vectors, axes=([0], [1]))

    ax.bar(x=np.arange(len(sim)), height=sim, width=0.8)
