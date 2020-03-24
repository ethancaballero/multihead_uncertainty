from utils.clb import L2


class MAS(L2):
    """
    @article{aljundi2017memory,
      title={Memory Aware Synapses: Learning what (not) to forget},
      author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
      booktitle={ECCV},
      year={2018},
      url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """

    def __init__(self, agent_config):
        super(MAS, self).__init__(agent_config)
        self.online_reg = True

    def calculate_importance(self, dataloader):
        self.log('Computing MAS')

        # Initialize the importance matrix
        if self.online_reg and len(self.regularization_terms)>0:
            importance = self.regularization_terms[1]['importance']
        else:
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized

        mode = self.training
        self.eval()

        # Accumulate the gradients of L2 loss on the outputs
        for i, (input, target, task) in enumerate(dataloader):
            if self.gpu:
                input = input.cuda()
                target = target.cuda()

            preds = self.forward(input)

            # Sample the labels for estimating the gradients
            # For multi-headed model, the batch of data will be from the same task,
            # so we just use task[0] as the task name to fetch corresponding predictions
            # For single-headed model, just use the max of predictions from preds['All']
            task_name = task[0] if self.multihead else 'All'

            # The flag self.valid_out_dim is for handling the case of incremental class learning.
            # if self.valid_out_dim is an integer, it means only the first 'self.valid_out_dim' dimensions are used
            # in calculating the  loss.
            pred = preds[task_name] if not isinstance(self.valid_out_dim, int) else preds[task_name][:,:self.valid_out_dim]

            pred.pow_(2)
            loss = pred.mean()

            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:  # Some heads can have no grad if no loss applied on them.
                    p += (self.params[n].grad.abs() / len(dataloader))

        self.train(mode=mode)

        return importance