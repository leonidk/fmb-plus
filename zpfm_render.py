import jax
import jax.numpy as jnp

def mrp_to_rot(vec):
    vec_mag = vec @ vec
    vec_mag_num = (1-vec_mag)
    vec_mag_den = ((1+vec_mag)**2)
    x,y,z = vec
    K = jnp.array(
           [[  0, -z,  y ],
            [  z,  0, -x ],
            [ -y,  x,  0 ]])
    R1 = jnp.eye(3) - ( ((4*vec_mag_num)/vec_mag_den) * K) + ((8/vec_mag_den) * (K @ K))
    R2 = jnp.eye(3)

    Rest = jnp.where(vec_mag > 1e-12,R1,R2)
    return Rest

def axangle_to_rot(axangl):
    scale = jnp.sqrt(axangl @ axangl)
    vec = axangl/scale
    x,y,z = vec
    K = jnp.array(
           [[  0, -z,  y ],
            [  z,  0, -x ],
            [ -y,  x,  0 ]])
    ctheta = jnp.cos(scale)
    stheta = jnp.sin(scale)
    R1 = jnp.eye(3) + stheta*K + (1-ctheta)*(K @ K)
    R2 = jnp.eye(3)
    Rest = jnp.where(scale > 1e-12,R1.T, R2)
    return Rest

def quat_to_rot(q):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z

    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    R1 = jnp.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
    R2 = jnp.eye(3)
    return jnp.where(Nq > 1e-12,R1,R2)


def render_func_rays(means, prec_full, weights_log, camera_starts_rays):
    prec = jnp.triu(prec_full)
    #weights = jnp.exp(weights_log)
    #weights = weights/weights.sum()

    def perf_idx(prcI,w,meansI):
        prc = prcI.T
        #prc = jnp.diag(jnp.sign(jnp.diag(prc))) @ prc
        div = jnp.prod(jnp.diag(jnp.abs(prc))) + 1e-20

        def perf_ray(r_t):
            r = r_t[0]
            t = r_t[1]
            p =  meansI -t 

            projp = prc @ p
            vsv = ((prc @ r)**2).sum()
            psv = ((projp) * (prc@r)).sum()
            projp2 = prc.T @ projp

            # linear
            res = (psv)/(vsv)
            
            v = r * res - p

            d0 = ((prc @ v)**2).sum()# + 3*jnp.log(jnp.pi*2)
            d2 = -0.5*d0 + w
            #d3 =  d2 + jnp.log(div) #+ 3*jnp.log(res)
            norm_est = projp2/jnp.linalg.norm(projp2)
            norm_est = jnp.where(r@norm_est < 0,norm_est,-norm_est)
            return res,d2,norm_est
        res,d2,projp  = jax.vmap((perf_ray))(camera_starts_rays) # jit perf
        return res, d2,projp

    zs,stds,projp = jax.vmap(perf_idx)(prec,weights_log,means)  # jit perf

    # compositing
    sample_density = jnp.exp(stds)  # simplier but splottier
    def sort_w(z,densities):
        idxs = jnp.argsort(z,axis=0)
        order_density = densities[idxs]
        order_summed_density = jnp.cumsum(order_density)
        order_prior_density =  order_summed_density - order_density
        ea = 1 - jnp.exp(-order_summed_density[-1])
        prior_density = jnp.zeros_like(densities)
        prior_density = prior_density.at[idxs].set(order_prior_density)
        transmit = jnp.exp(-prior_density)
        wout = transmit * (1-jnp.exp(-densities))
        return wout,ea
    w,est_alpha= jax.vmap(sort_w)(zs.T,sample_density.T)
    w = w.T

    wgt  = w.sum(0)
    div = jnp.where(wgt==0,1,wgt)
    w_n = w/div

    init_t=  (w_n*jnp.nan_to_num(zs)).sum(0)
    est_norm = (projp * w_n[:,:,None]).sum(axis=0)
    est_norm = est_norm/jnp.linalg.norm(est_norm,axis=1,keepdims=True)

    return init_t,est_alpha,est_norm,w

# axis angle rotations n * theta
def render_func_axangle(means, prec_full, weights_log, camera_rays, axangl, t):
    Rest = axangle_to_rot(axangl)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays)

# modified rod. parameters n * tan(theta/4)
def render_func_mrp(means, prec_full, weights_log, camera_rays, mrp, t):
    Rest = mrp_to_rot(mrp)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays)

# quaternions [cos(theta/2), sin(theta/2) * n]
def render_func_quat(means, prec_full, weights_log, camera_rays, quat, t):
    Rest = quat_to_rot(quat)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays)

def render_func_quat_cam(means, prec_full, weights_log, pixel_list, aspect, invF, quat, t):
    camera_rays = (pixel_list - jnp.array([0.5,0.5,0]))*jnp.array([invF,aspect*invF,1])

    Rest = quat_to_rot(quat)
    camera_rays = camera_rays @ Rest
    trans = jnp.tile(t[None],(camera_rays.shape[0],1))
    
    camera_starts_rays = jnp.stack([camera_rays,trans],1)
    return render_func_rays(means, prec_full, weights_log, camera_starts_rays)

def render_func_idx_quattrans(means, prec_full, weights_log, pixel_posei, invF, poses):
    rot_mats = jax.vmap(quat_to_rot)(poses[:,:4])
    def rot_ray_t(rayi):
        ray = rayi[:3] * jnp.array([invF,invF,1])
        pose_idx = rayi[3].astype(int)
        return jnp.array([ray@rot_mats[pose_idx],poses[pose_idx][4:]])
    camera_rays_start= jax.vmap(rot_ray_t)(pixel_posei)
    return render_func_rays(means, prec_full, weights_log, camera_rays_start)


def render_func_idx_quattrans_flow(means, prec_full, weights_log, pixel_posei, invF, poses):
    rot_mats = jax.vmap(quat_to_rot)(poses[:,:4])
    def rot_ray_t(rayi):
        ray = rayi[:3] * jnp.array([invF,invF,1])
        pose_idx = rayi[3].astype(int)
        return jnp.array([ray@rot_mats[pose_idx],poses[pose_idx][4:]])
    camera_rays_start = jax.vmap(rot_ray_t)(pixel_posei)

    est_depth,est_alpha,est_norm,est_w = render_func_rays(means, prec_full, weights_log, camera_rays_start)

    def flow_ray_i(rayi,depth):
        pose_idx = rayi[3].astype(int)
        pose_idxp1 = jax.lax.min(pose_idx+1,poses.shape[0]-1)
        pose_idxm1 = jax.lax.max(pose_idx-1,0)

        R1 = rot_mats[pose_idx]
        t1 = poses[pose_idx,4:]

        Rp1 = rot_mats[pose_idxp1]
        tp1 = poses[pose_idxp1,4:]

        ray = rayi[:3] * jnp.array([invF,invF,1])
        pt_cldc1 = ray * depth
        world_p = pt_cldc1 @ R1 + t1

        pt_cldc2 = (world_p- tp1) @ Rp1.T
        coord1 = pt_cldc2[:2]/(pt_cldc2[2]*invF)
        px_coordp = -(coord1 - rayi[:2])

        Rm1 = rot_mats[pose_idxm1]
        tm1 = poses[pose_idxm1,4:]

        pt_cldc3 = (world_p - tm1) @ Rm1.T
        coord2 = pt_cldc3[:2]/(pt_cldc3[2]*invF)
        px_coordm = -(coord2 - rayi[:2])

        return px_coordp,px_coordm

    flowp,flowm = jax.vmap(flow_ray_i)(pixel_posei,jnp.where(est_depth!=0,jnp.maximum(1e-7,est_depth),1e-7))

    return est_depth,est_alpha,est_norm,est_w,flowp,flowm


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
