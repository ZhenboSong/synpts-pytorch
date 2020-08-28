"""
parts of the code are modified from https://github.com/lvzhaoyang/DeeperInverseCompositionalAlgorithm
"""

import torch
import torch.nn.functional as F


def meshgrid(H, W, B=None, is_cuda=False):
    """ torch version of numpy meshgrid function

    :input
    :param height
    :param width
    :param batch size
    :param initialize a cuda tensor if true
    -------
    :return
    :param meshgrid in column
    :param meshgrid in row
    """
    u = torch.arange(0, W)
    v = torch.arange(0, H)

    if is_cuda:
        u, v = u.cuda(), v.cuda()

    u = u.repeat(H, 1).view(1, H, W)
    v = v.repeat(W, 1).t_().view(1, H, W)

    if B is not None:
        u, v = u.repeat(B, 1, 1, 1), v.repeat(B, 1, 1, 1)
    return u, v


def generate_xy_grid(B, H, W, K):
    """ Generate a batch of image grid from image space to world space
        px = (u - cx) / fx
        py = (y - cy) / fy

        function tested in 'test_geometry.py'

    :input
    :param batch size
    :param height
    :param width
    :param camera intrinsic array [fx,fy,cx,cy]
    ---------
    :return
    :param
    :param
    """
    fx, fy, cx, cy = K.split(1, dim=1)
    uv_grid = meshgrid(H, W, B)
    u_grid, v_grid = [uv.type_as(cx) for uv in uv_grid]
    px = ((u_grid.view(B, -1) - cx) / fx).view(B, 1, H, W)
    py = ((v_grid.view(B, -1) - cy) / fy).view(B, 1, H, W)
    return px, py


def visibilityEst(uv, z, k):
    '''
    estimate visibility using the local patches
     score: exp(-(z-z_min)^2/(z_max-z_min)^2)
     threshold: mean_score or 0.99
    :param uv: pixel coordinate [B, 2, N]
    :param z: depth coordinate [B, N]
    :param k: k nearest neighbour
    :return: visibility map [B, N]
    '''
    knn = KNN(k, transpose_mode=True)
    dist, indx = knn(uv, uv)  # B, N , k
    z_tor = z.unsqueeze(-1).repeat(1, 1, k)
    z_gather = torch.gather(z_tor, 1, indx)
    z_min = torch.min(z_gather, dim=2)[0]
    z_max = torch.max(z_gather, dim=2)[0]
    z_own = z_gather[:, :, 0]
    score = torch.exp(-(z_own-z_min)**2 / ((z_max-z_min)**2))  # B, N
    mean_score = torch.mean(score, dim=1)
    mask = score > mean_score
    return mask


def generateImage(uv, feature, sort, image_size):
    '''
    :param uv: pixel location in new image [B, 2. N]
    :param feature: projectiong features corresponding uv [B, C, N]
    :param sort: sorted descending depth index [B, N]
    :param image_size: new image size [H, W]
    :return: new image aligned with the feature as channel [B, C, H, W]
    '''
    B, C, N = feature.shape
    H, W = image_size

    if uv.is_cuda:
        uv_int = torch.round(uv).type(dtype=torch.cuda.LongTensor)
    else:
        uv_int = torch.round(uv).type(dtype=torch.LongTensor)
    u, v = torch.split(uv_int, 1, dim=1)

    # outlier
    inlier = (u >= 0) & (u < W) & (v >= 0) & (v < H)  # [B, N]
    u[~inlier] = 0
    v[~inlier] = 0
    feature[(~inlier).repeat(1, C, 1)] = 0

    # sort
    u_sort = torch.gather(u.transpose(1, 2), 1, sort.unsqueeze(-1))  # B, N, 1
    v_sort = torch.gather(v.transpose(1, 2), 1, sort.unsqueeze(-1))  # B, N, 1
    uv_ind = u_sort + v_sort * W

    f_sort = torch.gather(feature.transpose(1, 2), 1, sort.unsqueeze(-1).repeat(1, 1, C))  # B, N, C

    # image
    image = torch.ones((B, H * W, C), dtype=feature.dtype, device=feature.device).\
        scatter(1, uv_ind.repeat(1, 1, C), f_sort).transpose(1, 2)

    return image.view(B, C, H, W)


def projectPointFeature2Image(ptc, ptf, K, image_size):
    '''
    pad zeros to handle outliers, outliers have zero grad, pixels without depth point should be 1
    :param ptc: point3D [B, 3, N]
    :param ptf: point features [B, C, N]
    :param K:  [fx, fy, cx, cy]
    :param image_size: [H, W]
    :return:
    '''
    sort = torch.sort(ptc[:, 2, :], dim=-1, descending=True)[1]
    uv = project3Dto2D(ptc, K)
    B, C, N = ptf.shape
    H, W = image_size

    if uv.is_cuda:
        uv_int = torch.round(uv).type(dtype=torch.cuda.LongTensor)
    else:
        uv_int = torch.round(uv).type(dtype=torch.LongTensor)
    u, v = torch.split(uv_int, 1, dim=1)

    # outlier give -1 indicators to get zero grad
    inlier = (u >= 0) & (u < W) & (v >= 0) & (v < H)  # [B, 1, N]
    u[~inlier] = -1
    v[~inlier] = -1

    # sort
    u_sort = torch.gather(u.transpose(1, 2), 1, sort.unsqueeze(-1))+1  # B, N, 1
    v_sort = torch.gather(v.transpose(1, 2), 1, sort.unsqueeze(-1))+1  # B, N, 1
    uv_ind = u_sort + v_sort * (W+1)
    f_sort = torch.gather(ptf.transpose(1, 2), 1, sort.unsqueeze(-1).repeat(1, 1, C))  # B, N, C

    image = torch.zeros((B, (H+1) * (W+1), C), dtype=ptf.dtype, device=ptf.device, requires_grad=True).\
        scatter(1, uv_ind.repeat(1, 1, C), f_sort).transpose(1, 2).view(B, C, H+1, W+1)
    crop_image = image[:, :, 1:, 1:]
    return crop_image


def transform3Dto3D(xyz_tensor, T):
    '''
    transform the point cloud w.r.t. the transformation matrix
    :param xyz_tensor: [B, 3, N]
    :param T: tranformation T [B, 4, 4]
    '''
    B, _, N = xyz_tensor.size()
    # the transformation process is simply:
    t = T[:, 0:3, 3]  # B,3
    R = T[:, 0:3, 0:3]  # B,3,3
    xyz_t_tensor = torch.bmm(R, xyz_tensor) + t.unsqueeze(-1)  # B, 3, N

    return xyz_t_tensor


def project3Dto2D(xyz_tensor, K):
    """ Project a point cloud into pixels (u,v) given intrinsic K
    [u';v';w] = [K][x;y;z]
    u = u' / w; v = v' / w

    :param the xyz points [B, 3, N]
    :param calibration is a torch array composed of [fx, fy, cx, cy]
    -------
    :return u, v grid tensor in image coordinate
    """
    B, _, N = xyz_tensor.size()

    x, y, z = torch.split(xyz_tensor, 1, dim=1)
    fx, fy, cx, cy = torch.split(K, 1, dim=1)

    u = fx * x.squeeze(1) / z.squeeze(1) + cx
    v = fy * y.squeeze(1) / z.squeeze(1) + cy

    return torch.cat((u.unsqueeze(1), v.unsqueeze(1)), dim=1)


def projet2Dto3D(depth, K):
    """ project pixels (u,v) to a point cloud given intrinsic
    :param depth dim B*H*W
    :param calibration is torch array composed of [fx, fy, cx, cy]
    :param color (optional) dim B*3*H*W
    -------
    :return xyz tensor (batch of point cloud)
    (tested through projection)
    """
    if depth.dim() == 3:
        B, H, W = depth.size()
    else:
        B, _, H, W = depth.size()

    x, y = generate_xy_grid(B, H, W, K)
    z = depth.view(B, 1, H, W)
    pt3 = torch.cat((x * z, y * z, z), dim=1)

    return pt3.view(B, 3, -1)


def projet2Dto3D_resample(depth, K, scalor):
    """ project pixels (u,v) to a point cloud given intrinsic, and uniform resampling
    :param depth dim B*H*W
    :param calibration is torch array composed of [fx, fy, cx, cy]
    :param color (optional) dim B*3*H*W
    -------
    :return xyz tensor (batch of point cloud)
    (tested through projection)
    """
    if depth.dim() == 3:
        B, H, W = depth.size()
    else:
        B, _, H, W = depth.size()

    x, y = generate_xy_grid(B, H, W, K)
    z = depth.view(B, 1, H, W)
    pt3 = torch.cat((x * z, y * z, z), dim=1)
    pt3 = F.interpolate(pt3, scale_factor=scalor)

    return pt3.view(B, 3, -1)
