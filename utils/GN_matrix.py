import torch
from backpack import extend
from backpack.extensions import KFRA

from utils.backpack import backpack
from .nngeometry_old.maths import kronecker
from .nngeometry_old.pspace import M2Gradients
from .nngeometry_old.representations import AbstractMatrix
from .nngeometry_old.utils import get_individual_modules
from .nngeometry_old.vector import Vector


class M2Gradients_GN(M2Gradients):
    def __init__(self, model, dataloader, loss_function):
        super(M2Gradients_GN, self).__init__(model, dataloader, loss_function)
        self.loss_function = extend(self.loss_function)
        self.model = extend(self.model)

    def get_gn_blocks(self):
        for batch_idx, (data, target) in enumerate(data_loader):
            self.layers.zero_grad()
            output = self(data.view(len(data),-1).to(self.device))
            loss = self.criteria(output, target.to(self.device))
            with backpack(KFRA()):
                loss.backward()
        print("KFRA calculated")
        self._blocks = dict()
        for m in self.mods:
            sG = m.weight.size(0)
            mod_class = m.__class__.__name__
            if mod_class == 'Linear':
                sA = m.weight.size(1)
            elif mod_class == 'Conv2d':
                sA = m.weight.size(1) * m.weight.size(2) * m.weight.size(3)
            if m.bias is not None:
                sA += 1
            self._blocks[m] = (torch.zeros((sA, sA), device=device),
                               torch.zeros((sG, sG), device=device)

        for i,param in enumerate(self.layers.parameters()):
            if len(param.kfra) >1:
                self.kfra.append(param.kfra)


        return blocks



class GN_Matriix(AbstractMatrix):
    def __init__(self, generator):
        self.generator = generator
        self.data = generator.get_gn_blocks()

    def trace(self):
        return sum([torch.trace(a) * torch.trace(g) for a, g in self.data.values()])

    def get_matrix(self, split_weight_bias=False):
        """
        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        s = self.generator.get_n_parameters()
        M = torch.zeros((s, s), device=self.generator.get_device())
        mods, p_pos = get_individual_modules(self.generator.model)
        for mod in mods:
            a, g = self.data[mod]
            start = p_pos[mod]
            sAG = a.size(0) * g.size(0)
            if split_weight_bias:
                reconstruct = torch.cat([torch.cat([kronecker(g, a[:-1,:-1]), kronecker(g, a[:-1,-1:])], dim=1),
                                         torch.cat([kronecker(g, a[-1:,:-1]), kronecker(g, a[-1:,-1:])], dim=1)], dim=0)
                M[start:start+sAG, start:start+sAG].add_(reconstruct)
            else:
                M[start:start+sAG, start:start+sAG].add_(kronecker(g, a))
        return M

    def mv(self, vs):
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        for m in vs_dict.keys():
            v = vs_dict[m][0].view(vs_dict[m][0].size(0), -1)
            if m.bias is not None:
                v = torch.cat([v, vs_dict[m][1].unsqueeze(1)], dim=1)
            a, g = self.data[m]
            mv = torch.mm(torch.mm(g, v), a)
            if m.bias is None:
                mv_tuple = (mv,)
            else:
                mv_tuple = (mv[:, :-1].contiguous(), mv[:, -1:].contiguous())
            out_dict[m] = mv_tuple
        return Vector(model=vs.model, dict_repr=out_dict)

    def vTMv(self, vector):
        vector_dict = vector.get_dict_representation()
        norm2 = 0
        for mod in vector_dict.keys():
            v = vector_dict[mod][0].view(vector_dict[mod][0].size(0), -1)
            if len(vector_dict[mod]) > 1:
                v = torch.cat([v, vector_dict[mod][1].unsqueeze(1)], dim=1)
            a, g = self.data[mod]
            norm2 += torch.dot(torch.mm(torch.mm(g, v), a).view(-1), v.view(-1))
        return norm2

    def frobenius_norm(self):
        return sum([torch.trace(torch.mm(a, a)) * torch.trace(torch.mm(g, g))
                    for a, g in self.data.values()])**.5




