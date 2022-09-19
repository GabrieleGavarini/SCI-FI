import torch.utils.data

from OutputFeatureMapsManager import OutputFeatureMapsManager
from FaultInjectionExecutor import FaultInjectionExecutor

from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = resnet18(weights=ResNet18_Weights.DEFAULT)
    dataset = CIFAR10(root='./data',
                      train=False,
                      download=True,
                      transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    ofm_manager = OutputFeatureMapsManager(network=network,
                                           loader=loader,
                                           device=device)

    ofm_manager.save_intermediate_layer_outputs()

    fault_injection_executor = FaultInjectionExecutor(network=network,
                                                      loader=loader,
                                                      clean_output=ofm_manager.clean_output,
                                                      ofm=ofm_manager.output_feature_maps_dict)

    fault_injection_executor.run_faulty_campaign_on_weight([['conv1', 1, 1, 1, 1]])


if __name__ == '__main__':
    main()
