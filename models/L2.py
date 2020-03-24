from models.default import NormalNN
class L2(NormalNN):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """
    def __init__(self, agent_config):
        super(L2, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.regularization_terms = {}
        self.task_count = 0
        self.online_reg = True  # True: There will be only one importance matrix and previous model parameters
                                # False: Each task has its own importance matrix and model parameters

    def calculate_importance(self, dataloader):
        # Use an identity importance so it is an L2 regularization.
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)  # Identity
        return importance

    def learn_batch(self, train_loader, val_loader=None):

        self.log('#reg_term:', len(self.regularization_terms))

        # 1.Learn the parameters for current task
        super(L2, self).learn_batch(train_loader, val_loader)

        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance(train_loader)

        # Save the weight and importance of weights of current task
        self.task_count += 1
        if self.online_reg and len(self.regularization_terms)>0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {'importance':importance, 'task_param':task_param}
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {'importance':importance, 'task_param':task_param}

    def criterion(self, inputs, targets, tasks, regularization=True, **kwargs):
        loss = super(L2, self).criterion(inputs, targets, tasks, **kwargs)

        if regularization and len(self.regularization_terms)>0:
            # Calculate the reg_loss only when the regularization_terms exists
            reg_loss = 0
            for i,reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
            loss += self.config['reg_coef'] * reg_loss
        return loss