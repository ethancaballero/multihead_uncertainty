from utils.clb import L2


class SI(L2):
    """
    @inproceedings{zenke2017continual,
        title={Continual Learning Through Synaptic Intelligence},
        author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
        booktitle={International Conference on Machine Learning},
        year={2017},
        url={https://arxiv.org/abs/1703.04200}
    }
    """

    def __init__(self, agent_config):
        super(SI, self).__init__(agent_config)
        self.online_reg = True  # Original SI works in an online updating fashion
        self.damping_factor = 0.1
        self.w = {}
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()

        # The initial_params will only be used in the first task (when the regularization_terms is empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()

    def update_model(self, inputs, targets, tasks):

        unreg_gradients = {}

        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2. Collect the gradients without regularization term
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks, regularization=False)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for n, p in self.params.items():
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

        # 3. Normal update with regularization
        loss = self.criterion(out, targets, tasks, regularization=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= unreg_gradients[n] * delta  # w[n] is >=0

        return loss.detach(), out

    """
    # - Alternative simplified implementation with similar performance -
    def update_model(self, inputs, targets, tasks):
        # A wrapper of original update step to include the estimation of w
        # Backup prev param if not done yet
        # The backup only happened at the beginning of a new task
        if len(self.prev_params) == 0:
            for n, p in self.params.items():
                self.prev_params[n] = p.clone().detach()
        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()
        # 2.Calculate the loss as usual
        loss, out = super(SI, self).update_model(inputs, targets, tasks)
        # 3.Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if p.grad is not None:  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= p.grad * delta  # w[n] is >=0
        return loss.detach(), out
    """

    def calculate_importance(self, dataloader):
        self.log('Computing SI')
        assert self.online_reg, 'SI needs online_reg=True'

        # Initialize the importance matrix
        if len(self.regularization_terms) > 0:  # The case of after the first task
            importance = self.regularization_terms[1]['importance']
            prev_params = self.regularization_terms[1]['task_param']
        else:  # It is in the first task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized
            prev_params = self.initial_params

        # Calculate or accumulate the Omega (the importance matrix)
        for n, p in importance.items():
            delta_theta = self.params[n].detach() - prev_params[n]
            p += self.w[n] / (delta_theta ** 2 + self.damping_factor)
            self.w[n].zero_()

        return importance