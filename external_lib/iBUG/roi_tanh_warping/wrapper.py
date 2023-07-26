import math
import cv2
import torch
import numpy as np
import external_lib.iBUG.roi_tanh_warping.reference_impl as roi_warp_np
import external_lib.iBUG.roi_tanh_warping.pytorch_impl as roi_warp_pth
from vujade.vujade_debug import printd


class RoIWarpNumpy(object):
    def __init__(self,
                 _model: str = 'roi_tanh_polar',
                 _dsize_polar: tuple = (256, 256),
                 _angular_offset: float = math.pi * 0.0 / 180, # radian
                 _is_square: bool = False,
                 _is_nearest: bool = False,
                 _is_keep_aspect_ratio: bool = True) -> None:
        super(RoIWarpNumpy, self).__init__()

        '''
        Usage:
            i)  ndarr_polar = roiwarp_np.forward(_ndarr_cartesian=ndarr_img, _bbox_face=face_boxes[biggest_face_idx])
            ii) ndarr_carts = roiwarp_np.inverse(_ndarr_polar=ndarr_polar, _bbox_face=face_boxes[biggest_face_idx], _dsize_cartesian=(ndarr_img.shape[1::-1]))
        '''

        if not _model in {'roi_tanh_warp', 'roi_tanh_polar', 'roi_tanh_circular_warp', 'roi_tanh_polar_to_roi_tanh'}:
            raise ValueError('The {} is not supported.'.format(_model))
        else:
            self.model = _model

        self.dsize_polar = _dsize_polar
        self.angular_offset = _angular_offset
        self.is_square = _is_square
        self.interpolation = cv2.INTER_NEAREST if _is_nearest is True else cv2.INTER_LINEAR
        self.is_keep_aspect_ratio = _is_keep_aspect_ratio

        if self.model == 'roi_tanh_warp':
            self.func_forward = roi_warp_np.roi_tanh_warp
            self.func_inverse = roi_warp_np.roi_tanh_restore
        elif self.model == 'roi_tanh_polar':
            self.func_forward = roi_warp_np.roi_tanh_polar_warp
            self.func_inverse = roi_warp_np.roi_tanh_polar_restore
        elif self.model == 'roi_tanh_circular_warp':
            self.func_forward = roi_warp_np.roi_tanh_circular_warp
            self.func_inverse = roi_warp_np.roi_tanh_circular_restore
        elif self.model == 'roi_tanh_polar_to_roi_tanh':
            self.func_forward = roi_warp_np.roi_tanh_polar_to_roi_tanh
            self.func_inverse = roi_warp_np.roi_tanh_to_roi_tanh_polar
        else:
            raise NotImplementedError

    def forward(self, _ndarr_cartesian: np.ndarray, _bbox_face: np.ndarray):
        bbox_face = roi_warp_np.make_square_rois(_bbox_face[:4]) if self.is_square is True else _bbox_face

        if self.model in {'roi_tanh_polar', 'roi_tanh_circular_warp'}:
            res = self.func_forward(_ndarr_cartesian,
                                    bbox_face,
                                    self.dsize_polar[0],
                                    self.dsize_polar[1],
                                    angular_offset=self.angular_offset,
                                    border_mode=cv2.BORDER_REPLICATE,
                                    keep_aspect_ratio=self.is_keep_aspect_ratio)
        elif self.model in {'roi_tanh_warp'}:
            res = self.func_forward(_ndarr_cartesian,
                                    bbox_face,
                                    self.dsize_polar[0],
                                    self.dsize_polar[1],
                                    angular_offset=self.angular_offset,
                                    border_mode=cv2.BORDER_REPLICATE)
        else:
            raise NotImplementedError

        return res

    def inverse(self, _ndarr_polar: np.ndarray, _bbox_face: np.ndarray, _dsize_cartesian: tuple):
        bbox_face = roi_warp_np.make_square_rois(_bbox_face[:4]) if self.is_square is True else _bbox_face

        if self.model in {'roi_tanh_polar', 'roi_tanh_circular_warp'}:
            res = self.func_inverse(_ndarr_polar,
                                    bbox_face,
                                    _dsize_cartesian[0],
                                    _dsize_cartesian[1],
                                    angular_offset=self.angular_offset,
                                    interpolation=self.interpolation,
                                    border_mode=cv2.BORDER_REPLICATE,
                                    keep_aspect_ratio=self.is_keep_aspect_ratio)
        elif self.model in {'roi_tanh_warp'}:
            res = self.func_inverse(_ndarr_polar,
                                    bbox_face,
                                    _dsize_cartesian[0],
                                    _dsize_cartesian[1],
                                    angular_offset=self.angular_offset,
                                    interpolation=self.interpolation,
                                    border_mode=cv2.BORDER_REPLICATE)
        else:
            raise NotImplementedError

        return res


class RoIWarpPyTorch(object):
    def __init__(self,
                 _model: str = 'roi_tanh_polar',
                 _dsize_polar: tuple = (256, 256),
                 _angular_offset: float = math.pi * 0.0 / 180, # radian
                 _is_square: bool = False,
                 _is_nearest: bool = False,
                 _is_keep_aspect_ratio: bool = True) -> None:
        super(RoIWarpPyTorch, self).__init__()

        '''
        Usage:
            i)  tensor_polar = roiwarp_pt.forward(_tensor_cartesian=tensor_img, _bbox_face=tensor_bbox_face)
            ii) tensor_carts = roiwarp_pt.inverse(_tensor_polar=tensor_polar, _bbox_face=tensor_bbox_face, _dsize_cartesian=tensor_img.size()[:-3:-1])
        '''

        if not _model in {'roi_tanh_warp', 'roi_tanh_polar', 'roi_tanh_circular_warp', 'roi_tanh_polar_to_roi_tanh'}:
            raise ValueError('The {} is not supported.'.format(_model))
        else:
            self.model = _model

        self.dsize_polar = _dsize_polar
        self.angular_offset = _angular_offset
        self.is_square = _is_square
        self.interpolation = 'nearest' if _is_nearest is True else 'bilinear'
        self.is_keep_aspect_ratio = _is_keep_aspect_ratio

        if self.model == 'roi_tanh_warp':
            self.func_forward = roi_warp_pth.roi_tanh_warp
            self.func_inverse = roi_warp_pth.roi_tanh_restore
        elif self.model == 'roi_tanh_polar':
            self.func_forward = roi_warp_pth.roi_tanh_polar_warp
            self.func_inverse = roi_warp_pth.roi_tanh_polar_restore
        elif self.model == 'roi_tanh_circular_warp':
            self.func_forward = roi_warp_pth.roi_tanh_circular_warp
            self.func_inverse = roi_warp_pth.roi_tanh_circular_restore
        elif self.model == 'roi_tanh_polar_to_roi_tanh':
            self.func_forward = roi_warp_pth.roi_tanh_polar_to_roi_tanh
            self.func_inverse = roi_warp_pth.roi_tanh_to_roi_tanh_polar
        else:
            raise NotImplementedError

    def forward(self, _tensor_cartesian: torch.Tensor, _bbox_face: torch.Tensor):
        if (_tensor_cartesian.ndim != 4) or (_bbox_face.ndim != 2):
            raise NotImplementedError

        bbox_face = roi_warp_pth.make_square_rois(_bbox_face) if self.is_square is True else _bbox_face

        if self.model in {'roi_tanh_polar', 'roi_tanh_circular_warp'}:
            res = self.func_forward(_tensor_cartesian,
                                    bbox_face,
                                    self.dsize_polar[0],
                                    self.dsize_polar[1],
                                    angular_offsets=self.angular_offset,
                                    padding='border',
                                    keep_aspect_ratio=self.is_keep_aspect_ratio)
        elif self.model in {'roi_tanh_warp'}:
            res = self.func_forward(_tensor_cartesian,
                                    bbox_face,
                                    self.dsize_polar[0],
                                    self.dsize_polar[1],
                                    angular_offsets=self.angular_offset,
                                    padding='border')
        else:
            raise NotImplementedError

        return res

    def inverse(self, _tensor_polar: torch.Tensor, _bbox_face: torch.Tensor, _dsize_cartesian: tuple):
        bbox_face = roi_warp_pth.make_square_rois(_bbox_face) if self.is_square is True else _bbox_face

        if self.model in {'roi_tanh_polar', 'roi_tanh_circular_warp'}:
            res = self.func_inverse(_tensor_polar,
                                    bbox_face,
                                    _dsize_cartesian[0],
                                    _dsize_cartesian[1],
                                    angular_offsets=self.angular_offset,
                                    interpolation=self.interpolation,
                                    padding='border',
                                    keep_aspect_ratio=self.is_keep_aspect_ratio)
        elif self.model in {'roi_tanh_warp'}:
            res = self.func_inverse(_tensor_polar,
                                    bbox_face,
                                    _dsize_cartesian[0],
                                    _dsize_cartesian[1],
                                    angular_offsets=self.angular_offset,
                                    interpolation=self.interpolation,
                                    padding='border')
        else:
            raise NotImplementedError

        return res