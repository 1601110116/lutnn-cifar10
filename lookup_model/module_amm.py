import torch
import typing
import abc
from torch import nn
from torch import distributed
from lookup_model import clustering


class l2_square_similarity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, pts):
        # x: (N, nsubDs, subD, nblocks)
        # pts: (npts, nsubDs, subD)
        # sims: (N, npts, nsubDs, nblocks)
        N, _, _, nblocks = x.size()
        sims = - torch.cdist(
            x1=x.permute(1, 0, 3, 2).flatten(1, 2),
            x2=pts.permute(1, 0, 2)
        ).unflatten(dim=1, sizes=(N, nblocks)).permute(1, 3, 0, 2).square()
        return sims


class ModuleAMM(nn.Module):

    layer_id: str
    amm_config: typing.Dict
    base_module: nn.Module
    D: int
    M: int
    subD: int
    nsubDs: int
    nblocks: int
    pts: torch.Tensor
    temperature: torch.Tensor
    table: torch.Tensor
    get_similarity: typing.Callable
    clustering: typing.Callable
    space_divider: nn.Module
    x_unfolder: nn.Module
    weight_unfolder: nn.Module
    out_folder: nn.Module

    def __init__(self, amm_config, base_module: nn.Module):
        super().__init__()
        self.amm_config = amm_config
        self.base_module = base_module
        self.npts = amm_config['npts']
        self.need_lutnn_init = False

    def init_lut(self):
        self.nsubDs = int(self.D / self.subD)
        ref_param = self.base_module.weight
        self.pts = nn.Parameter(torch.zeros(
            (self.npts, self.D), dtype=ref_param.dtype, device=ref_param.device))
        temperature = torch.zeros((1, ), dtype=ref_param.dtype, device=ref_param.device)
        if self.amm_config['temperature'] > 0:
            temperature[0] = self.amm_config['temperature']
            self.temperature = nn.Parameter(temperature)
        else:
            temperature[0] = 1.0
            self.register_buffer('temperature', temperature)
        self.space_divider = nn.Unflatten(
            dim=1, unflattened_size=(self.nsubDs, self.subD))
        self.get_similarity = l2_square_similarity()
        self.clustering = clustering.kmeans_stochastic_relaxation

    @torch.no_grad()
    def update_pts_by_centroids_distributed(self, x):
        distributed.barrier()
        if distributed.get_rank() == 0:
            pts = torch.zeros(
                (self.npts, self.nsubDs, self.subD))
            x = self.space_divider(
                self.x_unfolder(x)).permute(1, 0, 3, 2).flatten(1, 2)
            for isubD in range(self.nsubDs):
                pts[:, isubD, :] = self.clustering(x[isubD])
            self.pts.copy_(pts.flatten(1, 2).to(x.device))
        distributed.barrier()
        distributed.broadcast(self.pts, src=0)
        print(f'layer {self.layer_id} got centroids of shape: {self.pts.size()}')

    @torch.no_grad()
    def update_pts_by_centroids(self, x):
        pts = torch.zeros(
            (self.npts, self.nsubDs, self.subD), dtype=x.dtype, device=x.device)
        x = self.space_divider(
            self.x_unfolder(x)).permute(1, 0, 3, 2).flatten(1, 2)
        for isubD in range(self.nsubDs):
            pts[:, isubD, :] = self.clustering(samples=x[isubD], npts=self.npts)
        self.pts.copy_(pts.flatten(1, 2))
        print(f'layer {self.layer_id} got centroids of shape: {self.pts.size()}')

    def build_table(self):
        pts = self.space_divider(self.pts)
        weight = self.space_divider(
            self.weight_unfolder(self.base_module.weight).flatten(1, 2))
        self.table = torch.einsum('pcd,mcd->pcm', pts, weight)

    def get_code(self, x: torch.Tensor, return_sims=False, keep_dim=False):
        sims = self.get_similarity(
            self.space_divider(self.x_unfolder(x)),
            self.space_divider(self.pts))
        codes = torch.argmax(sims, dim=1, keepdim=keep_dim)
        if return_sims:
            return sims, codes
        else:
            return codes

    @abc.abstractmethod
    def add_bias(self, out):
        pass

    def amm_forward(self, x):
        sims, codes = self.get_code(x, return_sims=True, keep_dim=True)
        hard_atts: torch.Tensor = torch.zeros_like(sims)
        hard_atts.scatter_(
            dim=1, index=codes,
            src=torch.ones_like(codes, dtype=sims.dtype, device=sims.device))
        soft_atts = torch.softmax(sims / self.temperature, dim=1)

        hard_out = torch.einsum('npcb,pcm->nmb', hard_atts, self.table)
        soft_out = torch.einsum('npcb,pcm->nmb', soft_atts, self.table)
        out = soft_out - (soft_out - hard_out).detach()

        # print(f'sims: {sims.size()}, codes: {codes.size()}, hard_atts: {hard_atts.size()}, out: {out.size()}, table: {self.table.size()}')

        out = self.out_folder(out)
        return self.add_bias(out)

    def fake_forward(self, x):
        x1 = self.space_divider(self.x_unfolder(x))
        weight1 = self.space_divider(self.weight_unfolder(self.base_module.weight).flatten(1, 2))
        out = torch.einsum('ncdb,mcd->nmb', x1, weight1)
        return self.out_folder(out)

    def forward(self, x):
        if self.need_lutnn_init:
            self.update_pts_by_centroids(x)
            out = self.base_module(x)
            self.need_lutnn_init = False
            return out
        self.build_table()
        out = self.amm_forward(x)
        return out


class Conv2dAMM(ModuleAMM):

    base_module: nn.Conv2d
    Hin: int
    Win: int
    Hout: int
    Wout: int

    def __init__(self, amm_config, base_module: nn.Conv2d):
        super().__init__(amm_config, base_module)
        if base_module.kernel_size[0] == 1:
            self.subD = self.amm_config['subD_k1']
        elif base_module.kernel_size[0] == 3:
            self.subD = self.amm_config['subD_k3']
        elif base_module.kernel_size[0] == 7:
            self.subD = self.amm_config['subD_k7']
        self.D = base_module.in_channels * base_module.kernel_size[0] * base_module.kernel_size[1]
        self.M = base_module.out_channels

        self.Hout = int(amm_config['Hin'] / base_module.stride[0])
        self.Wout = int(amm_config['Win'] / base_module.stride[1])
        self.nblocks = self.Hout * self.Wout
        self.init_lut()

    def init_lut(self):
        super().init_lut()
        self.x_unfolder = nn.Unfold(
            kernel_size=self.base_module.kernel_size,
            padding=self.base_module.padding,
            stride=self.base_module.stride)
        self.weight_unfolder = nn.Unfold(
            kernel_size=self.base_module.kernel_size)
        self.out_folder = nn.Fold(
            output_size=(self.Hout, self.Wout), kernel_size=1)

    def add_bias(self, out):
        if self.base_module.bias is not None:
            out.add_(self.base_module.bias.view(1, -1, 1, 1))
        return out


class LinearAMM(ModuleAMM):

    base_module: nn.Linear

    def __init__(self, amm_config, base_module: nn.Linear):
        super().__init__(amm_config, base_module)
        self.D = base_module.in_features
        self.M = base_module.out_features
        self.nblocks = 1
        self.init_lut()

    def init_lut(self):
        super().init_lut()
        self.x_unfolder = nn.Unflatten(dim=1, unflattened_size=(self.D, 1))
        self.weight_unfolder = nn.Unflatten(dim=1, unflattened_size=(self.D, 1))
        self.out_folder = nn.Flatten(1, 2)

    def add_bias(self, out):
        if self.base_module.bias is not None:
            out.add_(self.base_module.bias.view(1, -1))
        return out