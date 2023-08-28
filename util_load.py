import numpy as np
from pytorch3d.transforms import so3_log_map, matrix_to_quaternion

# wraps function call to save own history
# useful if using scipy.minimize but want to exit optimize whenever
class ExceptionWrap:
    def __init__(self,func):
        self.func = func
        self.results = []
    def __call__(self,*argv):
        f = self.func(*argv)
        self.results.append((f,tuple(argv[0])))
        return f
    
def load_mesh_with_pyt3d(shape_file,torch_device):
    from pytorch3d.io import load_obj, load_ply
    import pytorch3d.io as py3dIO
    from pytorch3d.structures import Meshes
    from iopath.common.file_io import PathManager
    import torch
    from pytorch3d.renderer import TexturesVertex

    if shape_file[-3:] == 'obj':
        #print('got obj')
        verts, faces_idx, _ = load_obj(shape_file)
        faces = faces_idx.verts_idx
    elif shape_file[-3:] == 'ply':
        #print('got ply')
        verts, faces = load_ply(shape_file)
    elif shape_file[-3:] == 'off':
        mesh2 = py3dIO.off_io.MeshOffFormat().read(shape_file,include_textures=False,
                                 device=torch_device,path_manager=PathManager())
        verts = mesh2.verts_list()[0]
        faces = mesh2.faces_list()[0]
    else:
        raise Exception("Not supported format")

    #verts = verts-verts.mean(0)
    #shape_scale = float(verts.std(0).mean())*3
    #verts = verts/shape_scale
        
    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(torch_device))

    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    mesh = Meshes(
        verts=[verts.to(torch_device)],   
        faces=[faces.to(torch_device)], 
        textures=textures
    )
    shape_scale = float(verts.std(0).mean())*3
    center = np.array(mesh.verts_list()[0].mean(0).detach().cpu())

    return mesh,shape_scale,center

def resize_2d_nonan(array,factor):
    """
    intial author: damo_ma
    """
    xsize, ysize = array.shape

    if isinstance(factor,int):
        factor_x = factor
        factor_y = factor
    elif isinstance(factor,tuple):
        factor_x , factor_y = factor[0], factor[1]
    else:
        raise NameError('Factor must be a tuple (x,y) or an integer')

    if not (xsize %factor_x == 0 or ysize % factor_y == 0) :
        raise NameError('Factors must be intger multiple of array shape')

    new_xsize, new_ysize = xsize//factor_x, ysize//factor_y

    new_array = np.empty([new_xsize, new_ysize])
    new_array[:] = np.nan # this saves us an assignment in the loop below

    # submatrix indexes : is the average box on the original matrix
    subrow, subcol  = np.indices((factor_x, factor_y))

     # new matrix indexs
    row, col  = np.indices((new_xsize, new_ysize))

    for i, j, ind in zip(row.reshape(-1), col.reshape(-1),range(row.size)) :
        # define the small sub_matrix as view of input matrix subset
        sub_matrix = array[subrow+i*factor_x,subcol+j*factor_y]
        # modified from any(a) and all(a) to a.any() and a.all()
        # see https://stackoverflow.com/a/10063039/1435167
        if (np.isnan(sub_matrix)).sum() < (factor_x*factor_y)/2.0 + (np.random.rand() -0.5): # if we haven't all NaN
            if (np.isnan(sub_matrix)).any(): # if we haven no NaN at all
                (new_array.reshape(-1))[ind] = np.nanmean(sub_matrix)
            else: # if we haven some NaN
                (new_array.reshape(-1))[ind] = np.mean(sub_matrix)
        # the case assign NaN if we have all NaN is missing due 
        # to the standard values of new_array

    return new_array

def convert_pyt3dcamera(cam, image_size):
    height, width = image_size
    cx = (width-1)/2
    cy = (height-1)/2
    f = (height/np.tan((np.pi/180)*float(cam.fov[0])/2))*0.5
    K = np.array([[f, 0, cx],[0,f,cy],[0,0,1]])
    pixel_list = (np.array(np.meshgrid(width-np.arange(width)-1,height-np.arange(height)-1,[0]))[:,:,:,0]).reshape((3,-1)).T

    camera_rays = (pixel_list - K[:,2])/np.diag(K)
    camera_rays[:,-1] = 1
    return np.array(camera_rays), np.array(so3_log_map(cam.R.cpu())[0]), np.array(-cam.R.cpu()[0]@cam.T.cpu()[0])