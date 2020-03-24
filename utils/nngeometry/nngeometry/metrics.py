import torch
import torch.nn.functional as F
from .pspace.m2gradients import M2Gradients


def FIM_MonteCarlo1(representation, loader, model,
                    variant='classif_logsoftmax'):
    """
    Helper to create a matrix computing the Fisher Information
    Matrix using a Monte-Carlo estimate with 1 sample per example
    """

    if variant == 'classif_logsoftmax':

        def loss(input, target):
            log_softmax = F.log_softmax(model(input), dim=1)
            probabilities = torch.exp(log_softmax)
            sampled_targets = torch.multinomial(probabilities, 1)
            return torch.gather(log_softmax, 1, sampled_targets)

        generator = M2Gradients(model=model,
                                dataloader=loader,
                                loss_function=loss)
        return representation(generator)
    else:
        raise NotImplementedError
