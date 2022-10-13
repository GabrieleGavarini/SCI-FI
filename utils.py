import os

import torch
import numpy as np
import pandas as pd


class NoChangeOFMException(Exception):
    """
    Exception thrown when the execution of a faulty neural network doesn't produce any difference between its own
    output feature map and the output feature map of a clean network execution
    """
    pass


def check_difference(check_control: bool,
                     golden: torch.Tensor,
                     faulty: torch.Tensor,
                     threshold: float):
    """
    If check control is true, check whether golden and faulty are the same. If faulty contains at least one nan, raise
    NoChangeOFMException. If no element of the faulty tensor has a distance from the same of element of the golden
    tensor greater than threshold, raise a NoChangeOFMException
    :param check_control: Whether to check the two tensors
    :param golden: The golden tensor
    :param faulty: The faulty tensor
    :param threshold: The threshold
    :return:
    """
    if check_control and torch.isnan(faulty).sum() == 0 and torch.sum((golden - faulty).abs() > threshold) == 0:
        raise NoChangeOFMException



def formatted_print(fault_list: list,
                    network_name: str,
                    batch_id: int,
                    faulty_prediction_dict: dict) -> None:
    """
    A function that prints to csv the results of the fault injection campaign on a single batch
    :param fault_list: A list of the faults
    :param network_name: The name of the network
    :param batch_id: The id of the batch
    :param faulty_prediction_dict: A dictionary where the key is the fault index and the value is a list of all the
    top_1 prediction for all the image of the given the batch
    """

    fault_list_rows = [[fault_id,
                       fault.layer_name,
                        fault.tensor_index[0],
                        fault.tensor_index[1] if len(fault.tensor_index) > 1 else np.nan,
                        fault.tensor_index[2] if len(fault.tensor_index) > 2 else np.nan,
                        fault.tensor_index[3] if len(fault.tensor_index) > 3 else np.nan,
                        fault.bit,
                        fault.value
                        ]
                       for fault_id, fault in enumerate(fault_list)
                       ]

    fault_list_columns = [
        'Fault_ID',
        'Fault_Layer',
        'Fault_Index_0',
        'Fault_Index_1',
        'Fault_Index_2',
        'Fault_Index_3',
        'Fault_Bit',
        'Fault_Value'
    ]

    prediction_rows = [
        [
            fault_id,
            batch_id,
            prediction_id,
            prediction
        ]
        for fault_id in faulty_prediction_dict for prediction_id, prediction in enumerate(faulty_prediction_dict[fault_id])
    ]

    prediction_columns = [
        'Fault_ID',
        'Batch_ID',
        'Image_ID',
        'Top_1'
    ]

    fault_list_df = pd.DataFrame(fault_list_rows, columns=fault_list_columns)
    prediction_df = pd.DataFrame(prediction_rows, columns=prediction_columns)

    complete_df = fault_list_df.merge(prediction_df, on='Fault_ID')

    os.makedirs(f'output/fault_campaign_results/{network_name}', exist_ok=True)
    complete_df.to_csv(f'output/fault_campaign_results/{network_name}/fault_injection_batch_{batch_id}.csv', index=False)
