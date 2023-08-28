import optax
from util import DegradeLR
import fm_render

import jax
import jax.numpy as jnp
def shape_objective(params,pixel_list,invF,poses,beta2,beta3,beta4,reg_amt,true_alpha):
    CLIP_ALPHA = 1e-6
    flat_alpha = true_alpha.ravel()
    mean,prec,weight_log = params
    render_res = fm_render.render_func_idx_quattrans_flow(mean,prec,weight_log,pixel_list,invF,poses,beta2,beta3,beta4)

    est_alpha = render_res[2]
    est_alpha = jnp.clip(est_alpha,CLIP_ALPHA,1-CLIP_ALPHA)
    mask_loss = - ((flat_alpha * jnp.log(est_alpha)) + (1-flat_alpha)*jnp.log(1-est_alpha))

    est_depth = render_res[0]
    xdiff = (jnp.diff(est_depth.reshape(true_alpha.shape),axis=0,append=0)**2).ravel()
    ydiff = (jnp.diff(est_depth.reshape(true_alpha.shape),axis=1,append=0)**2).ravel()

    reg = jnp.where(flat_alpha > 0.5,xdiff+ydiff,0)
    return mask_loss.mean() + reg_amt*reg.mean()

def pose_objective(poses,mean,prec,weight_log,pixel_list,invF,beta2,beta3,beta4,true_alpha,true_fwd,true_bwd,flow_mul):
    CLIP_ALPHA = 1e-6
    render_res = fm_render.render_func_idx_quattrans_flow(mean,prec,weight_log,pixel_list,invF,poses,beta2,beta3,beta4)

    est_alpha = render_res[2]
    est_alpha = jnp.clip(est_alpha,CLIP_ALPHA,1-CLIP_ALPHA)
    mask_loss = - ((true_alpha * jnp.log(est_alpha)) + (1-true_alpha)*jnp.log(1-est_alpha))

    pad_alpha = true_alpha[:,None]
    flow1 = jnp.abs(pad_alpha*true_fwd.reshape((-1,2))-pad_alpha*render_res[5])
    flow2 = jnp.abs(pad_alpha*true_bwd.reshape((-1,2))-pad_alpha*render_res[6])
    return mask_loss.mean() + flow_mul*(flow1.mean() + flow2.mean())

def shape_pose_objective(params,pixel_list,beta2,beta3,beta4,true_alphas):
    CLIP_ALPHA = 1e-6
    mean,prec,weight_log,invF,poses = params

    def eval_frame(pose,true_alpha):
        flat_alpha = true_alpha.ravel()
        render_res = fm_render.render_func_idx_quattrans_flow(mean,prec,weight_log,pixel_list,invF,poses,beta2,beta3,beta4)

        est_alpha = render_res[2]
        est_alpha = jnp.clip(est_alpha,CLIP_ALPHA,1-CLIP_ALPHA)
        mask_loss = - ((flat_alpha * jnp.log(est_alpha)) + (1-flat_alpha)*jnp.log(1-est_alpha))
        return mask_loss.mean()
    per_frames = jax.vmap(eval_frame)(poses,true_alphas)
    return per_frames.mean()

def reconstruct_shape(vg_objective,init_shape,degrade_settings,render_settings,pose,Niter,reg_amt,reference):

    # babysit learning rates
    adjust_lr = DegradeLR(*degrade_settings)
    beta2,beta3,beta4,pixel_list,invF = render_settings

    optimizer = optax.adam(adjust_lr.step_func)
    opt_state = optimizer.init(init_shape)

    params = init_shape

    losses = []
    for i in range(Niter):
        val,g = vg_objective(params,pixel_list,invF,pose,beta2,beta3,beta4,reg_amt,reference)
        val = float(val)
        losses.append(val)
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        if adjust_lr.add(val):
            break
    return params,losses

def obtain_pose(vg_objective,shape,degrade_settings,render_settings,poses,Niter,reference,fwdflow,bwdflow,flow_amt):

    # babysit learning rates
    adjust_lr = DegradeLR(*degrade_settings)
    beta2,beta3,beta4,pixel_list,invF = render_settings
    mean, prec, weight_log = shape

    optimizer = optax.sgd(adjust_lr.step_func,0.95)
    opt_state = optimizer.init(poses)
    params = poses
    losses = []
    for i in range(Niter):
        val,g = vg_objective(params,mean, prec, weight_log,pixel_list,invF,beta2,beta3,beta4,reference,fwdflow,bwdflow,flow_amt)
        val = float(val)
        losses.append(val)
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)

        if adjust_lr.add(val):
            break
    return params,losses

def refine_shapepose(vg_objective,init_shape,degrade_settings,render_settings,poses,Niter,references):
    # babysit learning rates
    adjust_lr = DegradeLR(*degrade_settings)
    beta2,beta3,beta4,pixel_list,invF = render_settings

    optimizer = optax.adam(adjust_lr.step_func,0.9)
    init_state = list(init_shape) + [invF]+ [poses]
    opt_state = optimizer.init(init_state)

    losses = []
    loop = range(Niter)

    params = init_state
    
    for i in loop:
        val,g = vg_objective(params,pixel_list,beta2,beta3,beta4,references[:len(poses)])
     
        val = float(val)
        losses.append(val)
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)

        if adjust_lr.add(val):
            break
    return params,losses

def readFlow(fn):
    import numpy as np
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))