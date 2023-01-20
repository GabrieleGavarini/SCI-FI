import torch
from torch import Tensor
from torch.nn import Module


class InjectableOutputModule(Module):

    def __init__(self,
                 clean_module: Module,
                 device: torch.device,
                 layer_name: str,
                 input_size: torch.Size = None,
                 output_size: torch.Size = None):
        super().__init__()

        # --- PRIVATE PARAMETERS --- #

        # The device where to run the injection
        self.__device = device

        # The clean module to replace
        self.__clean_module = clean_module

        # The mask of the clean output
        self.__output_clean_mask = None

        # The value of the fault and the related mask
        self.__output_fault_mask = None
        self.__output_fault = None

        # --- PUBLIC PARAMETERS --- #

        self.layer_name = layer_name
        self.input_size = input_size
        self.output_size = output_size


    def inject_fault(self,
                     output_fault: Tensor,
                     output_fault_mask: Tensor = None) -> None:
        """
        Inject a fault in the output feature map of the layer
        :param output_fault: the fault to add to the clean output of the module
        :param output_fault_mask: Default None. If not specified, the mask is automatically created from the
        output_fault: the mask contains zero where the output_fault is zero and one otherwise
        """
        self.__output_fault = output_fault

        # The fault mask can be implicitly computed from the fault output
        if output_fault_mask is None:
            self.__output_fault_mask = self.__output_fault.ne(0).int()
        else:
            self.__output_fault_mask = output_fault_mask

        # The clean_mask is 1 - fault_mask. Basically, it needs to zero all the values affected by a fault in the output
        self.__output_clean_mask = (torch.ones(size=self.__output_fault_mask.shape, device=self.__device) - self.__output_fault_mask).int()


    def clean_fault(self):
        """
        Remove an injected fault from the output feature map of the layer
        """

        self.__output_fault_mask = None


    def forward(self,
                input_tensor: Tensor) -> Tensor:
        """
        The forward function of an injectable module execute the module forward function without applying any additional
        values if the module is not affected by any fault. Otherwise, it sums the clean mask times the clean output +
        the fault output * fault mask:
            output = mask_clean * output_clean + mask_fault * output_fault
        :param input_tensor: The input tensor of the module
        :return: The module forward affected by a fault is the module is affected by a fault
        """

        output_clean = self.__clean_module(input_tensor)

        if self.__output_fault_mask is not None:

            masked_output_clean = torch.mul(output_clean, self.__output_clean_mask)
            masked_output_fault = torch.mul(self.__output_fault, self.__output_fault_mask)

            return masked_output_clean + masked_output_fault
        else:
            return output_clean
