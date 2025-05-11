import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1] # 2

    if grid_dim == 1:
        zeros = torch.zeros_like(coords)
        coords = torch.cat([coords,zeros],dim=-1)
        grid_dim +=1
        
    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)    # [N, 2] → [1, N, 2]

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    # 原始坐标形状: [B, N, 2]
    # 转换后形状: [B, 1, ..., 1, N, 2] (中间插入 grid_dim-1 个1) [B,1,N,2]
    # grid_sample 要求坐标张量形状为 [B, H, W, grid_dim]
    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    # sample后interp形状：[B,feature_dim,1,N]
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs

def init_composite_grid(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    
    """初始化复合网格参数，包含三种组合类型"""
    params = nn.ParameterList()
    
    # 类型1: 传统二维平面组合 (6个)
    coo_combs = list(itertools.combinations(range(4), 2))
    for dims in coo_combs:  # XY, XZ, YZ等
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in dims[::-1]]
        ))
        if 3 in dims:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)

        params.append(new_grid_coef)
    
    # 类型2: 一维时间组合 (4个)
    # 一维向量看作（N，1）的二维向量
    for dim in range(4):  # X,Y,Z,T单独
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[dim],1]
        ))
        if dim==3:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        params.append(new_grid_coef)
    
    # 类型3: 1平面+2向量组合 (6组)
    for plane_dims in coo_combs:  # XY, XZ, YZ平面
        new_plane_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in plane_dims[::-1]]
        ))
        if 3 in plane_dims:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
            
        params.append(new_plane_coef)
        
        for i in range(4):
            if i not in plane_dims:
                new_vector_coef = nn.Parameter(torch.empty(
                    [1, out_dim] + [reso[i],1]
                ))
                if i==3:  # Initialize time planes to 1
                    nn.init.ones_(new_vector_coef)
                else:
                    nn.init.uniform_(new_vector_coef, a=a, b=b)
                params.append(new_vector_coef)
                
    return params

def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

def extract_composite_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            ms_mlps: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(4), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
        
    grid: nn.ParameterList
    
    # 多分辨率特征累积
    combined_feat = 0
    for level,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        features = []
        
        # 类型1: 二维平面特征
        for ci, coo_comb in enumerate(coo_combs):
            i = ci + 0
            feature_dim = grid[i].shape[1]
            interp_out_plane = (
                grid_sample_wrapper(grid[i], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # 将六平面值累积
            interp_space = interp_space * interp_out_plane
        features.append(interp_space)   
            
        # 类型2: 一维向量特征
        interp_space = 1.
        for ci in range(4):
            i = ci + 6
            feature_dim = grid[i].shape[1]
            interp_out_plane = (
                grid_sample_wrapper(grid[i], pts[..., [ci]])
                .view(-1, feature_dim)
            )
            # 将4个一维向量值累积
            interp_space = interp_space * interp_out_plane
        features.append(interp_space)   
        
        # 类型3：1个二维平面+2个一维向量
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            i = ci*3 + 10
            feature_dim = grid[i].shape[1]
            
            # 二维平面XY
            interp_out_plane = (
                grid_sample_wrapper(grid[i], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            
            # 一维向量Z、T
            interp_out_vectors = 1.
            num = 1
            for cor in range(4):
                if cor not in coo_comb:
                    interp_out_vector = (
                        grid_sample_wrapper(grid[i+num], pts[..., [cor]])
                        .view(-1, feature_dim)
                    )
                    num +=1
                    interp_out_vectors = interp_out_vectors * interp_out_vector
            
            # 将平面值和向量值累积
            interp_space = interp_space * interp_out_plane * interp_out_vectors
        features.append(interp_space)  
        
        # 拼接所有特征
        level_feat = torch.cat(features, dim=-1)
        combined_feat += ms_mlps[level](level_feat) * (0.5**level)  # 高层级权重衰减

    return combined_feat

'''
         kplanes_config = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 32,
            'resolution': [64, 64, 64, 75]
            }
            bounds=1.6
            multires=[1, 2]
            empty_voxel=False
            static_mlp=False
        '''
class HexPlaneField(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. 初始化网格和MLP
        self.mlps = nn.ModuleList()
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            # gp = init_grid_param(
            #     grid_nd=config["grid_dimensions"],
            #     in_dim=config["input_coordinate_dim"],
            #     out_dim=config["output_coordinate_dim"],
            #     reso=config["resolution"],
            # )
            
            # 生成三种组合类型的网格参数
            gp = init_composite_grid(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            
            # shape[1] is out-dim - Concatenate over feature len for each scale
            # if self.concat_features:
            #     self.feat_dim += gp[-1].shape[1]
            # else:
            #     self.feat_dim = gp[-1].shape[1]
                
            self.feat_dim += gp[-1].shape[1]
            
            self.grids.append(gp)
            
        for grid in self.grids:
            input_dim = grid[-1].shape[1]
            
            # 为每个分辨率层级创建MLP
            mlp = nn.Sequential(
                nn.Linear(input_dim*3, input_dim*3),
                nn.ReLU(),
                nn.Linear(input_dim*3, self.feat_dim)
            )
            self.mlps.append(mlp)
            
        print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
        
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        
        features = extract_composite_features(
            pts, ms_grids=self.grids,  # noqa
            ms_mlps=self.mlps,
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)

        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features
