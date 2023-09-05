import re
from ast import literal_eval as make_tuple

from typing import List, Tuple, Type

def convert_index(matrix_shape, index):
    """
    Convert the given index to the corresponding indices for a matrix of the specified shape.

    :param matrix_shape: The shape of the matrix in the order (M, N, P, Q).
    :type matrix_shape: tuple
    :param index: The index to be converted.
    :type index: int

    :return: A tuple containing the corresponding indices for each dimension (M, N, P, Q).
    :rtype: tuple

    :raises AssertionError: If any of the calculated indices exceed the size of their respective dimensions.
    """

    indices = []

    for dimension_size in reversed(matrix_shape):
        index, remainder = divmod(index, dimension_size)
        indices.insert(0, remainder)

    return tuple(indices)

    # if len(matrix_shape) == 2:
    #     M, N = matrix_shape
    #     m, n = divmod(index, N)
    #     return m,n
    # else:
    #     M, N, P, Q = matrix_shape
    #     index, q = divmod(index, Q)
    #     index, p = divmod(index, P)
    #     m, n = divmod(index, N)
    #     assert m < M and n < N and p < P and q < Q
    #     return m, n, p, q

def get_list_of_tuples_from_str(string: str,
                                  element_in_tuple: int = 3) -> List[Tuple]:
    """
    Get a list of tuples from a string
    :param string: The string to convert
    :param element_in_tuple: How many elements are in a single tuple
    :return: A list of tuples
    """
    return [make_tuple(match[0]) for match in re.findall(f'(\(([0-9]+(, )?){{{element_in_tuple}}}\))', string)]


def get_list_from_str(string: str,
                        cast_type: Type = float) -> List:
    """
    Convert a string in a list of elements of type cast_type
    :param string: The string to convert
    :param cast_type: The type to cast the elements
    :return: The list
    """
    return [cast_type(entry) for entry in string.replace('[', '').replace(']', '').split(',')]