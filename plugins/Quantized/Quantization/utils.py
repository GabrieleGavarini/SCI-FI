from tqdm import tqdm
import numpy as np

import torch.nn


def evaluate(network: torch.nn.Module,
             loader: torch.utils,
             device: torch.device,
             desc: str = None) -> float:
    """
    Evaluate the target network over the specific data loader. Returns the average accuracy
    :param network: The network to be evaluated
    :param loader: The loader to be used
    :param device: The device where the network is loader
    :param desc: Default None. The description to append to the pbar
    :return: The accuracy of the network over the dataset
    """

    accuracy = 0
    total_predicted = 0
    top_1_predicted = 0

    if desc is None:
        desc = 'Testing the accuracy of the network'

    pbar = tqdm(loader,
                desc=desc)

    for batch in pbar:
        data, label = batch
        data = data.to(device)
        label = label.to(device)

        # Make predictions over the target dataset
        predictions = network(data)
        predicted_top_1 = torch.argmax(predictions, dim=-1)
        top_1_predicted += np.sum((predicted_top_1 == label).detach().cpu().numpy())
        total_predicted += len(label)

        # Measure teh accuracy
        accuracy = top_1_predicted/total_predicted
        pbar.set_postfix({'Accuracy': f'{100*accuracy:.2f}%'})

    return accuracy