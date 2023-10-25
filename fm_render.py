import jax
import jax.numpy as jnp

# this file implements most of https://arxiv.org/abs/2308.14737

# contains various rotation conversions
from util_render import *

# core rendering function
def render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3):
    # precision is fully parameterized by triangle matrix
    # we use upper triangle for compatibilize with sklearn
    prec = jnp.triu(prec_full)

    # if doing normalized weights for proper GMM
    # typically not used for shape reconstruction
    #weights = jnp.exp(weights_log)
    #weights = weights/weights.sum()

    # gets run per gaussian with [precision, log(weight), mean]
    def perf_idx(prcI,w,meansI):
        # math is easier with lower triangle
        prc = prcI.T

        # gaussian scale
        # could be useful for log likelihood but not used here
        div = jnp.prod(jnp.diag(jnp.abs(prc))) + 1e-20
        
        # gets run per ray
        def perf_ray(r_t):
            # unpack the ray (r) and position (t)
            r = r_t[0]
            t = r_t[1]
            
            # shift the mean to be relative to ray start
            p =  meansI - t

            # compute \sigma^{-0.5} p, which is reused
            projp = prc @ p

            # compute v^T \sigma^{-1} v
            vsv = ((prc @ r)**2).sum()

            # compute p^T \sigma^{-1} v
            psv = ((projp) * (prc@r)).sum()

            # compute the surface normal as \sigma^{-1} p
            projp2 = prc.T @ projp

            # distance to get maximum likelihood point for this gaussian
            # scale here is based on r! 
            # if r = [x, y, 1], then depth. if ||r|| = 1, then distance
            res = (psv)/(vsv)
            
            # get the intersection point
            v = r * res - p

            # compute intersection's unnormalized Gaussian log likelihood
            d0 = ((prc @ v)**2).sum()# + 3*jnp.log(jnp.pi*2)
            
            # multiply by the weight
            d2 = -0.5*d0 + w
            
            # if you wanted real probability
            #d3 =  d2 + jnp.log(div) #+ 3*jnp.log(res)

            # compute a normalized normal
            norm_est = projp2/jnp.linalg.norm(projp2)
            norm_est = jnp.where(r@norm_est < 0,norm_est,-norm_est)

            # return ray distance, gaussian distance, normal
            return res, d2, norm_est
        
        # runs parallel for each ray across each gaussian
        res,d2,projp  = jax.vmap((perf_ray))(camera_starts_rays)

        return res, d2,projp
    
    # runs parallel for gaussian
    zs,stds,projp = jax.vmap(perf_idx)(prec,weights_log,means) 

    # alpha is based on distance from all gaussians
    est_alpha = 1-jnp.exp(-jnp.exp(stds).sum(0) )

    # points behind camera should be zero
    # BUG: est_alpha should also use this
    sig1 = (zs > 0)# sigmoid
    
    # compute the algrebraic weights in the paper
    w = sig1*jnp.nan_to_num(jax_stable_exp(-zs*beta_2 + beta_3*stds))+1e-20

    # normalize weights
    wgt  = w.sum(0)
    div = jnp.where(wgt==0,1,wgt)
    w = w/div

    # compute weighted z and normal
    init_t =  (w*jnp.nan_to_num(zs)).sum(0)
    est_norm = (projp * w[:,:,None]).sum(axis=0)
    est_norm = est_norm/jnp.linalg.norm(est_norm,axis=1,keepdims=True)

    # return z, alpha, normal, and the weights
    # weights can be used to compute color, DINO features, or any other per-Gaussian property
    return init_t,est_alpha,est_norm,w

# renders image if rotation is in: axis angle rotations n * theta
def render_func_axangle(means, prec_full, weights_log, camera_rays, axangl, t, beta_2, beta_3):
    Rest = axangle_to_rot(axangl)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3,)

# renders image if rotation is in: modified rod. parameters n * tan(theta/4)
def render_func_mrp(means, prec_full, weights_log, camera_rays, mrp, t, beta_2, beta_3):
    Rest = mrp_to_rot(mrp)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3)

# renders image if rotation is in: quaternions [cos(theta/2), sin(theta/2) * n]
def render_func_quat(means, prec_full, weights_log, camera_rays, quat, t, beta_2, beta_3):
    Rest = quat_to_rot(quat)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3)

# renders image if rotation is quaternions and we have pixels with a single parameter inverse focal length
def render_func_quat_cam(means, prec_full, weights_log, pixel_list, aspect, invF, quat, t, beta_2, beta_3):
    camera_rays = (pixel_list - jnp.array([0.5,0.5,0]))*jnp.array([invF,aspect*invF,1])

    Rest = quat_to_rot(quat)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays, beta_2, beta_3)

# renders batch of rays
# takes pixel coords (pixels, pose)
# takes inverse focal length
# takes full set of poses as (translation, quaternion) pairs
def render_func_idx_quattrans(means, prec_full, weights_log, pixel_posei, invF, poses, beta_2, beta_3):
    rot_mats = jax.vmap(quat_to_rot)(poses[:,:4])
    def rot_ray_t(rayi):
        ray = rayi[:3] * jnp.array([invF,invF,1])
        pose_idx = rayi[3].astype(int)
        return jnp.array([ray@rot_mats[pose_idx],poses[pose_idx][4:]])
    camera_rays_start= jax.vmap(rot_ray_t)(pixel_posei)
    return render_func_rays(means, prec_full, weights_log, camera_rays_start, beta_2, beta_3)

# renders a batch of rays, as above, but also computes fwd & backward flow for each pixel. 
def render_func_idx_quattrans_flow(means, prec_full, weights_log, pixel_posei, invF, poses, beta_2, beta_3):
    rot_mats = jax.vmap(quat_to_rot)(poses[:,:4])
    def rot_ray_t(rayi):
        ray = rayi[:3] * jnp.array([invF,invF,1])
        pose_idx = rayi[3].astype(int)
        return jnp.array([ray@rot_mats[pose_idx],poses[pose_idx][4:]])
    camera_rays_start = jax.vmap(rot_ray_t)(pixel_posei)

    # render the pixels
    est_depth,est_alpha,est_norm,est_w = render_func_rays(means, prec_full, weights_log, camera_rays_start, beta_2, beta_3)

    # per pixel, compute the flow
    def flow_ray_i(rayi,depth):
        # find the pose index
        pose_idx = rayi[3].astype(int)
        pose_idxp1 = jax.lax.min(pose_idx+1,poses.shape[0]-1)
        pose_idxm1 = jax.lax.max(pose_idx-1,0)

        # get the pose
        R1 = rot_mats[pose_idx]
        t1 = poses[pose_idx,4:]

        # get the next pose
        Rp1 = rot_mats[pose_idxp1]
        tp1 = poses[pose_idxp1,4:]

        # compute this pose 3D
        ray = rayi[:3] * jnp.array([invF,invF,1])
        pt_cldc1 = ray * depth
        world_p = pt_cldc1 @ R1 + t1

        # transform and project back into next camera
        pt_cldc2 = (world_p- tp1) @ Rp1.T
        coord1 = pt_cldc2[:2]/(pt_cldc2[2]*invF)
        px_coordp = -(coord1 - rayi[:2])

        # get the previous pose
        Rm1 = rot_mats[pose_idxm1]
        tm1 = poses[pose_idxm1,4:]

        # transform and project back into previous camera
        pt_cldc3 = (world_p - tm1) @ Rm1.T
        coord2 = pt_cldc3[:2]/(pt_cldc3[2]*invF)
        px_coordm = -(coord2 - rayi[:2])

        return px_coordp, px_coordm

    # per ray, compute the flow
    flowp,flowm = jax.vmap(flow_ray_i)(pixel_posei,est_depth)

    return est_depth,est_alpha,est_norm,est_w,flowp,flowm

# just a log likelihood function
# in case you want to maximize 3D points
# verify that the above math/structure matches typical GMM log likelihood
# compare with sklearn
def log_likelihood(params, points):
    means, prec_full, weights_log = params
    prec = jnp.triu(prec_full)
    weights = jnp.exp(weights_log)
    weights = weights/weights.sum()

    def perf_idx(prcI,w,meansI):
        prc = prcI.T
        div = jnp.prod(jnp.diag(jnp.abs(prc)))

        def perf_ray(pt):
            p =  meansI -pt

            pteval = ((prc @ p)**2).sum()

            d0 = pteval+ 3*jnp.log(jnp.pi*2)
            d2 = -0.5*d0 + jnp.log(w)
            d3 =  d2 + jnp.log(div) 

            return d3
        res = jax.vmap((perf_ray))(points) # jit perf
        return res

    res = jax.vmap(perf_idx)(prec,weights,means)  # jit perf
    
    return -jax.scipy.special.logsumexp(res.T, axis=1).ravel().mean(),res# + ent.mean()
