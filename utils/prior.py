import time
import torch
from copy import copy, deepcopy
from utils.nngeometry.nngeometry.representations import DiagMatrix
from utils.nngeometry.nngeometry.utils import get_individual_modules
from utils.nngeometry.nngeometry.vector import PVector
from utils.nngeometry.nngeometry.metrics import FIM_MonteCarlo1


#diag, kfac, ekfak
class GaussianPrior(object):
    def __init__(self, model, train_loader, reg_matrix="diag", shuffle=False):
        self.shuffle=shuffle
        self.reg_matrix = reg_matrix
        if reg_matrix == "diag":
            from utils.nngeometry.nngeometry.representations import DiagMatrix as Reg_Matrix
        elif reg_matrix=="kfac":
            from utils.nngeometry.nngeometry.representations import KFACMatrix as Reg_Matrix
        elif reg_matrix=="ekfac":
            from utils.nngeometry.nngeometry.representations import EKFACMatrix as Reg_Matrix
        elif reg_matrix=="block_diag":
            from utils.nngeometry.nngeometry.representations import BlockDiagMatrix as Reg_Matrix

        print("Calculating Fisher "+ reg_matrix)
        self.model = model
        self.F_ = FIM_MonteCarlo1(representation=Reg_Matrix,
                                 loader=train_loader,
                                 model=self.model)
        if reg_matrix == "ekfac":
            self.F_.update_diag()
            diag = []
            for k, d in self.F_.diags.items():
                diag.append(d)
            diag_ = torch.abs(torch.cat(diag))
            F_ = DiagMatrix(generator=self.F_.generator,data=diag_)
            self.F_ = F_
        #elif reg_matrix=="kfac":
            #self.F_.evecs = dict()
            #self.F_.diags = dict()
            #self.F_.compute_eigendecomposition()

        #self.prev_params = PVector.from_model(self.model).clone().detach()
        n_parameters = self.F_.generator.get_n_parameters()
        print(str(n_parameters) + ' parameters')
        print("Done calculating curvature matrix")

        #v0 = PVector.from_model(self.model).get_dict_representation()
        self.prev_params = PVector.from_model(self.model).clone().detach()
            #PVector(model=self.model, vector_repr=deepcopy(
            #PVector(model=self.model, dict_repr=v0).get_flat_representation().detach()))

        #v0
        if self.shuffle:
            if isinstance(self.F_.data, dict):
                for key in self.F_.data.keys():
                    for item in self.F_.data[key]:
                        idx = torch.randperm(item.nelement())
                        item.data = item.view(-1)[idx].view(item.size())
            else:
                idx = torch.randperm(self.F_.data.nelement())
                self.F_.data = self.F_.data.view(-1)[idx].view(self.F_.data.size())


    def consolidate(self, new_prior, task):
        self.prev_params = PVector.from_model(new_prior.model).clone().detach()

        if self.reg_matrix=="ekfac":
            if isinstance(self.F_, DiagMatrix):
                self.F_.data = ((deepcopy(new_prior.F_.data)) + self.F_.data * (task)) / (task + 1)

            elif isinstance(self.F_.data, dict):
                for (n, p), (n_, p_) in zip(self.F_.data.items(),new_prior.F_.data.items()):
                    for item, item_ in zip(p, p_):
                        item.data = ((item.data*(task))+deepcopy(item_.data))/(task+1) #+ self.F_.data[n]
            #self.F_.generator.dataloader = new_prior.F_.generator.dataloader
            #self.F_.calc_eigendecomp()
            #self.F_.update_diag()

        else:
            if isinstance(self.F_.data, dict):
                for (n, p), (n_, p_) in zip(self.F_.data.items(),new_prior.F_.data.items()):
                    for item, item_ in zip(p, p_):
                        item.data = ((item.data*(task))+deepcopy(item_.data))/(task+1) #+ self.F_.data[n]
                #if self.reg_matrix == "kfac":
                #    self.F_.calc_eigendecomp()

            else:
                self.F_.data = ((deepcopy(new_prior.F_.data)) + self.F_.data*(task))/(task+1)
            if self.shuffle:
                if isinstance(self.F_.data, dict):
                    for key in self.F_.data.keys():
                        for item in self.F_.data[key]:
                            idx = torch.randperm(item.nelement())
                            item.data = item.view(-1)[idx].view(item.size())
                else:
                    idx = torch.randperm(self.F_.data.nelement())
                    self.F_.data = self.F_.data.view(-1)[idx].view(self.F_.data.size())

    def regularizer(self, model):
        params0_vec =  PVector(model=self.model, vector_repr=PVector.from_model(model).get_flat_representation())
        #v_1 = PVector.from_model(mod el).get_dict_representation()
        #params0_vec = PVector(model=self.model, vector_repr=PVector(model=model, dict_repr=v_1).get_flat_representation())
        v = params0_vec - self.prev_params
        # print(v.get_flat_representation())
        reg = self.F_.vTMv(v)
        # print(reg)
        return reg

