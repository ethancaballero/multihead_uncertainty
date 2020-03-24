import time
import torch
from copy import copy, deepcopy
from utils.nngeometry_old.pspace import M2Gradients
from utils.nngeometry_old.vector import Vector, from_model

class GaussianPrior(object):
    def __init__(self, model, train_loader, reg_matrix="diag", shuffle=False):
        self.shuffle=shuffle
        if reg_matrix == "diag":
            from utils.nngeometry_old.representations import DiagMatrix as Reg_Matrix
        elif reg_matrix=="kfac":
            from utils.nngeometry_old.representations import KFACMatrix as Reg_Matrix
        print("Calculating Fisher "+ reg_matrix +" matrix")

        self.model = deepcopy(model)
        m2_generator = M2Gradients(model=self.model, dataloader=train_loader, loss_function=self.model.loss_fim_mc_estimate)
        n_parameters = m2_generator.get_n_parameters()
        print(str(n_parameters) + ' parameters')
        self.F_ = Reg_Matrix(m2_generator)
        print("Done calculating curvature matrix")
        v0 = from_model(self.model)
        self.prev_params = Vector(model=self.model, vector_repr=deepcopy(Vector(model=self.model, dict_repr=v0).get_flat_representation().detach()))

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
        v0 = from_model(new_prior.model)
        self.prev_params = Vector(model=new_prior.model, vector_repr=Vector(model=new_prior.model, dict_repr=v0).get_flat_representation().detach_())

        if isinstance(self.F_.data, dict):
            for (n, p), (n_, p_) in zip(self.F_.data.items(),new_prior.F_.data.items()):
                for item, item_ in zip(p, p_):
                    item.data = ((item.data*(task))+deepcopy(item_.data))/(task+1) #+ self.F_.data[n]
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
        v_1 = from_model(model)
        params0_vec = Vector(model=self.model, vector_repr=Vector(model=model, dict_repr=v_1).get_flat_representation())
        v = params0_vec - self.prev_params
        #print(v.get_flat_representation())
        reg = self.F_.vTMv(v)
        #print(reg)
        return reg

