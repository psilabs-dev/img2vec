from typing import List, Tuple, TypeAlias, Union
import PIL
import PIL.Image
import numpy
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

FloatArrayT: TypeAlias = numpy.typing.NDArray[numpy.float64]
ImageOrImageListT: TypeAlias = Union[List[PIL.Image.Image], PIL.Image.Image]

class Img2VecException(Exception):
    def __init__(self, devices: List[str]):
        self.message = f"No such devices found: {','.join(devices)}"
        super().__init__(self.message)

class Img2Vec:
    RESNET_OUTPUT_SIZES = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048
    }

    EFFICIENTNET_OUTPUT_SIZES = {
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
        'efficientnet_b5': 2048,
        'efficientnet_b6': 2304,
        'efficientnet_b7': 2560
    }

    model: torch.nn.Module = None

    def __init__(
            self, model: str='resnet-18', layer: Union[str, int]='default', 
            layer_output_size: int=512, device_preference: List[str]=["cpu"]
    ):
        """ Img2Vec
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        :param device_preference: List[str] list of devices in order of preference (e.g. ["cuda", "cpu"]).
        """

        device: torch.device
        found_device = False
        for device_name in device_preference:
            match device_name:
                case "cpu":
                    found_device = True
                case "cuda":
                    found_device = torch.cuda.is_available()
                case "mps":
                    found_device = torch.backends.mps.is_available()
            if found_device:
                device = torch.device(device_name)
                break
        if not found_device:
            raise Img2VecException(device_preference)

        self.device = device
        self.layer_output_size = layer_output_size
        self.model_name = model
        self.layer = layer
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def is_downloaded(self) -> bool:
        return hasattr(self, 'model') and (self.model is not None)

    def download_model(self) -> "Img2Vec":
        if not self.is_downloaded():
            self.model, self.extraction_layer = self._get_model_and_layer(self.model_name, self.layer)
            self.model = self.model.to(device=self.device)
            self.model.eval()
        return self

    def close(self):
        if isinstance(self.model, torch.nn.Module):
            del self.model

    def get_vec(
            self, img: ImageOrImageListT, tensor: bool=False
    ) -> Union[FloatArrayT, torch.Tensor]:
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if not self.is_downloaded():
            raise TypeError("Model is not loaded!")
        if isinstance(img, list):
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            if self.model_name in ['alexnet', 'vgg']:
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            elif self.model_name == 'densenet' or 'efficientnet' in self.model_name:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                _ = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['alexnet', 'vgg']:
                    return my_embedding.numpy()[:, :]
                elif self.model_name == 'densenet' or 'efficientnet' in self.model_name:
                    return torch.mean(my_embedding, (2, 3), True).numpy()[:, :, 0, 0]
                else:
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

            if self.model_name in ['alexnet', 'vgg']:
                my_embedding = torch.zeros(1, self.layer_output_size)
            elif self.model_name == 'densenet' or 'efficientnet' in self.model_name:
                my_embedding = torch.zeros(1, self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                _ = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['alexnet', 'vgg']:
                    return my_embedding.numpy()[0, :]
                elif self.model_name == 'densenet':
                    return torch.mean(my_embedding, (2, 3), True).numpy()[0, :, 0, 0]
                else:
                    return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name: str, layer: Union[str, int]) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        model: torch.nn.Module

        if model_name.startswith('resnet') and not model_name.startswith('resnet-'):
            weights: torchvision.models.Weights
            match int((model_name[6:]).strip()):
                case 18:
                    weights = torchvision.models.ResNet18_Weights.DEFAULT
                case 34:
                    weights = torchvision.models.ResNet34_Weights.DEFAULT
                case 50:
                    weights = torchvision.models.ResNet50_Weights.DEFAULT
                case 101:
                    weights = torchvision.models.ResNet101_Weights.DEFAULT
                case 152:
                    weights = torchvision.models.ResNet152_Weights.DEFAULT
                case _:
                    raise ValueError(f"Invalid ResNet layer count: {model_name}")
            model = getattr(models, model_name)(weights=weights)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
            else:
                layer = model._modules.get(layer)
            return model, layer
        elif model_name == 'resnet-18':
            model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'vgg':
            # VGG-11
            model = models.vgg11_bn(weights=torchvision.models.VGG11_BN_Weights.DEFAULT)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = model.classifier[-1].in_features # should be 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'densenet':
            # Densenet-121
            model = models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
            if layer == 'default':
                layer = model.features[-1]
                self.layer_output_size = model.classifier.in_features # should be 1024
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer

        elif "efficientnet" in model_name:
            # efficientnet-b0 ~ efficientnet-b7
            b_number = int(model_name[14:].strip())
            match b_number:
                case 0:
                    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
                case 1:
                    weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
                case 2:
                    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
                case 3:
                    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
                case 4:
                    weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
                case 5:
                    weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
                case 6:
                    weights = torchvision.models.EfficientNet_B6_Weights.DEFAULT
                case 7:
                    weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT
                case _:
                    raise ValueError("Unsupported EfficientNet Architecture.")
            model_name = f"efficientnet_b{b_number}"
            model = getattr(models, model_name)(weights=weights)

            if layer == 'default':
                layer = model.features
                self.layer_output_size = self.EFFICIENTNET_OUTPUT_SIZES[model_name]
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)

if __name__ == "__main__":
    Img2Vec(device_preference=["cuda", "cpu"])
