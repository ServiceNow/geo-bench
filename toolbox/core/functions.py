import torch
from toolbox.core.task_specs import TaskSpecifications

def head_generator(task_specs, hyperparams):
    """
    Returns a an appropriate head based on the task specifications. We can use task_specs.task_type as follow:
        classification: 2 layer MLP with softmax activation
        semantic_segmentation: U-Net decoder.
    we can also do something special for a specific dataet using task_specs.dataset_name. Hyperparams and input_shape
    can also be used to adapt the head.

    Args:
        task_specs: object of type TaskSpecifications providing information on what type of task we are solving
        hyperparams: dict of hyperparameters.
        input_shape: list of tuples describing the shape of the input of this module. TO BE DISCUSSED: should this be
            the input itself? should it be a dict of shapes?
    """
    if task_specs.task_type == "classification":
        if hyperparams["head_type"] == "linear":
            (in_ch,) = task_specs.features_shape
            out_ch = task_specs.n_classes
            return torch.nn.Linear(in_ch, out_ch)
        else:
            raise ValueError(f"Unrecognized head type: {hyperparams['head_type']}")
    else:
        raise ValueError(f"Unrecognized task: {task_specs.task_type}")


def vit_head_generator(task_specs, hyperparams, input_shape):
    """
    ViT architectures may require different type of heads. In which case, we should provide this to the users as well. TO BE DISCUSSED.
    """
    pass


def train_loss_generator(task_specs, hyperparams):
    """
    Returns the appropriate loss function depending on the task_specs. We should implement basic loss and we can leverage the
    following attributes: task_specs.task_type and task_specs.eval_loss
    """
    if task_specs.task_type == "classification":
        if hyperparams["loss_type"] == "crossentropy":
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unrecognized loss type: {hyperparams['head_type']}")
    else:
        raise ValueError(f"Unrecognized task: {task_specs.task_type}")


def hparams_to_string(list_of_hp_configs):
    """
    Introspect the list of hyperparms configurations to find the hyperparameters that changes during the experiment e.g.,
    there might be 8 hyperparameters but only 2 that changes. For each hyperparameter that changes
    """


def compute_metrics(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f"accuracy-{k}"] = correct_k.mul_(100.0 / batch_size)
        return res
