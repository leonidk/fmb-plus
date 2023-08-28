
import matplotlib.pyplot as plt


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
    vmin = None,
    vmax = None,
    cmap = None,
    interp = 'nearest',
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw)
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3],interpolation=interp)
        else:
            # only render Alpha channel
            ax.imshow(im[...],vmin=vmin,vmax=vmax,cmap=cmap,interpolation=interp)
        if not show_axes:
            ax.set_axis_off()
    plt.tight_layout()


from scipy.special import erf
import numpy as np
class DegradeLR:
    def __init__(self, init_lr, p_thresh=5e-2, window=10, p_window=5, slope_less=0, max_drops = 4, print_debug=True):
        assert( (init_lr >0) and (p_thresh > 0) and (p_thresh < 1))
        self.init_lr = init_lr
        self.p_thresh = p_thresh
        self.window = int(round(window))
        if self.window < 3:
            print('window too small! clipped to 3')
            self.window = 3
        self.slope_less = slope_less
        self.p_window = int(round(p_window))
        if self.p_window < 1:
            print('p_window too small! clipped to 1')
            self.p_window = 1
        self.train_val = []
        self.prior_p = []
        self.n_drops = 0
        self.max_drops = max_drops
        self.last_drop_len = self.window+1
        self.step_func = lambda x: self.init_lr/(10** self.n_drops)
        self.print_debug = print_debug
        self.counter = 0
    def add(self,error):
        self.counter += 1
        self.train_val.append(error)
        len_of_opt = len(self.train_val)
        
        if len_of_opt >= self.window + self.p_window:
            yo = np.array(self.train_val[-self.window:])
            yo = yo/yo.mean()
            xo = np.arange(self.window)
            xv = np.vstack([xo,np.ones_like(xo)]).T
            w = np.linalg.pinv(xv.T @ xv) @ xv.T @ yo
            yh = xo*w[0] + w[1]
            var =((yh-yo)**2).sum() / (self.window-2)
            var_slope = (12*var)/(self.window**3)
            ps = 0.5*(1+ erf((self.slope_less-w[0])/(np.sqrt(2*var_slope))))
            self.prior_p.append(ps)
            
            p_eval = np.array(self.prior_p[-self.p_window:])
            if (p_eval < self.p_thresh).all():
                self.n_drops += 1
                if self.n_drops > self.max_drops:
                    if self.print_debug: 
                        print('early exit due to max drops')
                    return True
                if self.print_debug:
                    print('dropping LR to {:.2e} after {} steps'.format(self.step_func(0),self.counter-1))
                min_len = self.window+self.p_window
                if self.last_drop_len == min_len and len_of_opt == min_len:
                    if self.print_debug: 
                        print('early exit due to no progress')
                    return True
                self.last_drop_len = len(self.train_val)
                self.train_val = []
        return False

import jax.numpy as jnp
def compute_normals(camera_rays, depth_py_px, eps=1e-20):
    PY,PX = depth_py_px.shape
    nan_depth = jnp.nan_to_num(depth_py_px.ravel())
    dpt = jnp.array( camera_rays.reshape((-1,3)) * nan_depth[:,None] )
    dpt = dpt.reshape((PY,PX,3))
    ydiff = dpt - jnp.roll(dpt,1,0)
    xdiff = dpt - jnp.roll(dpt,1,1)
    ddiff = jnp.cross(xdiff.reshape((-1,3)),ydiff.reshape((-1,3)),)
    nan_ddiff = jnp.nan_to_num(ddiff,nan=1e-6)
    norms = nan_ddiff/(eps+jnp.linalg.norm(nan_ddiff,axis=1,keepdims=True))

    return norms
