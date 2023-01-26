from typing import Type

import torch


class NoChangeOFMException(Exception):
    """
    Exception thrown when the execution of a faulty neural network doesn't produce any difference between its own
    output feature map and the output feature map of a clean network execution
    """
    pass


def get_delayed_start_module_subclass(superclass_type: Type) -> Type:
    """
    Return the class dynamically extended from the module class type
    :param superclass_type: The type of the superclass, used to extend it
    :return:
    """

    # Define a DelayedStartModule class that dynamically extends the delayed_start_module_class to support an
    # overloading of the forward method, while being able to call the parent forward method
    class DelayedStartModule(superclass_type):

        def forward(self,
                    input_tensor: torch.Tensor) -> torch.Tensor:
            """
            Smart forward used for fault delayed start. With this smart function, the inference starts from the first layer
            marked as starting layer and the input of that layer is loaded from disk
            :param input_tensor: The module input tensor
            :return: The module output tensor
            """

            # If the starting layer and starting module are set, proceed with the smart forward
            if self.starting_layer is not None and self.starting_module is not None:

                # Execute the layers iteratively, starting from the one where the fault is injected
                layer_index = self.layers.index(self.starting_layer)

                # Create a dummy input
                # TODO: self.starting_layer.input_size
                x = torch.zeros(size=self.starting_module.input_size, device='cuda')

                # Specify that the first module inside this layer should load the input from memory and not read from previous
                # layer
                self.starting_module.start_from_this_layer()

                # Iteratively execute modules in the layer
                for layer in self.layers[layer_index:]:
                    x = layer(x)

                # Clear the marking on the first module
                self.starting_module.do_not_start_from_this_layer()

            # Otherwise, used the original forward function of the network
            else:
                x = super().forward(input_tensor)

            return x

    return DelayedStartModule
