import torch


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

    # DEBUG
    import time
    elapsed = 0
    for _ in range(1000):
        start_time = time.time()
        check_control and torch.all(faulty.eq(golden))
        elapsed += time.time() - start_time
    elapsed = elapsed / 1000
    print(f'Mean time {elapsed:.7f}')

    elapsed = 0
    for _ in range(1000):
        start_time = time.time()
        check_control and torch.sum((golden - faulty).abs()) == 0
        elapsed += time.time() - start_time
    elapsed = elapsed / 1000
    print(f'Mean time {elapsed:.7f}')

    if threshold == 0:
        if check_control and torch.sum((golden - faulty).abs()) == 0:
            raise NoChangeOFMException

    elif check_control and torch.sum((golden - faulty).abs() > threshold) == 0:
        raise NoChangeOFMException
