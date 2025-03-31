import numpy as np
import py360convert


def set_range_minus180to180(x):
    if -180 <= x <= 180:
        y = x
    elif 180 < x <= 540:
        y = x - 360
    else:
        raise ValueError("x is expected from -180 to 540")
    return y


def fold_back_azimuth(x):
    if -180 <= x < -90:
        y = -180 - x
    elif -90 <= x <= 90:
        y = x
    elif 90 < x <= 180:
        y = 180 - x
    else:
        raise ValueError("x is expected from -180 to 180")
    return y


class E2PFast(object):
    def __init__(self, in_hw, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear'):
        '''
        fov_deg: scalar or (scalar, scalar) field of view in degree
        u_deg:   horizon viewing angle in range [-180, 180]
        v_deg:   vertical viewing angle in range [-90, 90]
        '''
        h, w = in_hw

        try:
            h_fov, v_fov = fov_deg[0] * np.pi / 180, fov_deg[1] * np.pi / 180
        except:
            raise NotImplementedError('not implemented')
        in_rot = in_rot_deg * np.pi / 180

        if mode == 'bilinear':
            self.order = 1
        elif mode == 'nearest':
            self.order = 0
        else:
            raise NotImplementedError('unknown mode')

        u = -u_deg * np.pi / 180
        v = v_deg * np.pi / 180
        xyz = py360convert.utils.xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
        uv = py360convert.utils.xyz2uv(xyz)
        self.coor_xy = py360convert.utils.uv2coor(uv, h, w)

    def e2p(self, e_img):
        '''
        e_img:   ndarray in shape of [H, W, *]
        '''
        assert len(e_img.shape) == 3

        pers_img = np.stack([
            py360convert.utils.sample_equirec(e_img[..., i], self.coor_xy, order=self.order)
            for i in range(e_img.shape[2])
        ], axis=-1)

        return pers_img


def check_on_off(deg, azi, ele, out_height, out_width):
    # viewing angle
    out_u_deg = deg
    out_v_deg = 0

    # perspective video params
    out_height = out_height
    out_width = out_width
    out_u_fov_deg = 100  # field of view
    out_v_fov_deg = 2 * np.arctan(np.tan((np.pi / 180) * (out_u_fov_deg / 2)) * out_height / out_width) / np.pi * 180

    equi = np.zeros((180, 360, 3))  # related to elevation and azimuth range
    az = (180 - azi) % 360  # dcase labels range az (180, -180), el (90, -90). Convert to (0, 359) (0, 179)
    el = (90 - ele) % 180
    equi[el, az, 0] = 1

    planar = py360convert.e2p(equi, fov_deg=(out_u_fov_deg, out_v_fov_deg), u_deg=out_u_deg, v_deg=out_v_deg, out_hw=(out_height, out_width))

    # value: 1 for onscreen event, 0 for offscreen event
    # x, y:  position of onscreen event in perspective video
    if np.sum(planar) > 0:  # onscreen
        value = 1
        x = np.round(np.sum(np.sum(planar[:, :, 0], axis=0) * np.arange(out_width))  / np.sum(np.sum(planar[:, :, 0], axis=0)))
        y = np.round(np.sum(np.sum(planar[:, :, 0], axis=1) * np.arange(out_height)) / np.sum(np.sum(planar[:, :, 0], axis=1)))
    else:  # offscreen
        value = 0
        x, y = None, None
    return value, x, y


class CheckOnOffFixedDeg(object):
    def __init__(self, deg, out_height, out_width):
        # elevation and azimuth range
        self.in_height = 180
        self.in_width = 360

        # viewing angle
        self.out_u_deg = deg
        self.out_v_deg = 0

        # perspective video params
        self.out_height = out_height
        self.out_width = out_width
        self.out_u_fov_deg = 100  # field of view
        self.out_v_fov_deg = 2 * np.arctan(np.tan((np.pi / 180) * (self.out_u_fov_deg / 2)) * self.out_height / self.out_width) / np.pi * 180

        self.e2p_fast = E2PFast(in_hw=(self.in_height, self.in_width),
                                fov_deg=(self.out_u_fov_deg, self.out_v_fov_deg),
                                u_deg=self.out_u_deg, v_deg=self.out_v_deg,
                                out_hw=(self.out_height, self.out_width))

    def check(self, azi, ele):
        equi = np.zeros((180, 360, 3))  # related to elevation and azimuth range
        az = (180 - azi) % 360  # dcase labels range az (180, -180), el (90, -90). Convert to (0, 359) (0, 179)
        el = (90 - ele) % 180
        equi[el, az, 0] = 1

        planar = self.e2p_fast.e2p(equi)

        # value: 1 for onscreen event, 0 for offscreen event
        # x, y:  position of onscreen event in perspective video
        if np.sum(planar) > 0:  # onscreen
            value = 1
            x = np.round(np.sum(np.sum(planar[:, :, 0], axis=0) * np.arange(self.out_width))  / np.sum(np.sum(planar[:, :, 0], axis=0)))
            y = np.round(np.sum(np.sum(planar[:, :, 0], axis=1) * np.arange(self.out_height)) / np.sum(np.sum(planar[:, :, 0], axis=1)))
        else:  # offscreen
            value = 0
            x, y = None, None
        return value, x, y


class CheckOnOffFast(object):
    def __init__(self):
        self.aziele2valuexy_0deg = np.load("./aziele2valuexy_0deg_ufov100w640h360.npy")
    
    def check(self, deg, azi, ele):
        azi_0deg = (azi + deg) % 360
        value, x, y = self.aziele2valuexy_0deg[azi_0deg, ele]  # value: 1 for onscreen, 0 for offscreen
        return value, x, y
