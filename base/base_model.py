import torch
import torch.nn as nn
import torchsummary
from abc import abstractmethod
from vujade import vujade_flops_counter as flops_counter_
from vujade.vujade_debug import printf


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    def __str__(self) -> str:
        """
        Model prints with number of trainable parameters
        """
        params = self.count_parameters()
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    @abstractmethod
    def forward(self, *inputs) -> None:
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def _init_weights(self, _nonlinearity: str = 'leaky_relu') -> None:
        printf('The model name: {}.'.format(self.__class__.__name__), _is_pause=False)
        printf('The weights of model are initialized.', _is_pause=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if _nonlinearity == 'leaky_relu':
                    nn.init.kaiming_normal_(m.weight, a=1, mode='fan_in', nonlinearity=_nonlinearity)
                elif _nonlinearity == 'relu':
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                else:
                    raise NotImplementedError('The _nonlinearity, {} in the _init_weights() has not been supported yet.'.format(_nonlinearity))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_weights_kaming_normal(self, _scale: float = 1.0, _a: int = 0, _mode: str = 'fan_in', _nonlinearity: str = 'leaky_relu') -> None:
        """
        Principle:
            1) Output feature map should be gaussian distribution with zero-mean and small standard deviation
               in order to alleviate gradient vanishing problem for s-curve activation function such as the sgimoid.
               It makes sure the training is stable.
        Xavier initialization:
            1) The Xavier initialization method initializes weights to be gaussian distribution with zero-mean and 1/sqrt(n_input).
            2) It is recommend that the method use for the sigmoid and Tanh activation function.
            3) Please do not use the initialization method for the ReLU activation function
               because the mean of output feature maps tends to converge to 0.
               Thus, when using the ReLU activation function, please use the Kaming He initialization.
            4) Formula expressed in numpy: w = np.random.randn(n_input, n_output) / math.sqrt(n_input)
        Kaming He initialization:
            1) The Xavier initialization method initialize weights to be gaussian distribution with zero-mean and 1/sqrt(n_input/2).
            2) It is recommend that the method use for the ReLU activation function.
            3) Formula expressed in numpy: w = np.random.randn(n_input, n_output) / math.sqrt(n_input/2)
        Summary:
            1) Xavier initialization: sigmoid and Tanh activation function
            2) Kaming He initialization: ReLU activation function
        """
        printf('The model name: {}.'.format(self.__class__.__name__), _is_pause=False)
        printf('The weights of model are initialized.', _is_pause=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=_a, mode=_mode, nonlinearity=_nonlinearity)
                m.weight.data *= _scale  # For residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # Option 1
                nn.init.kaiming_normal_(m.weight, a=_a, mode=_mode, nonlinearity=_nonlinearity)
                m.weight.data *= _scale  # For residual block
                # Option 2
                # m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def summary(self, _input_shape: tuple, _batch_size: int = 1, _device: str = 'cpu', _is_summary: bool = True) -> str:
        if not _device in {'cpu', 'gpu'}:
            raise ValueError

        if isinstance(_input_shape, tuple) is False:
            _input_shape = tuple(_input_shape)

        printf('{} Network summary.'.format(self.__class__.__name__), _is_pause=False)

        if _is_summary is True:
            torchsummary.summary(self, input_size=_input_shape, batch_size=_batch_size, device=_device)

        tensor_input = torch.randn([1, *_input_shape], dtype=torch.float).to(_device)
        counter = flops_counter_.add_flops_counting_methods(self)
        counter.eval().start_flops_count()
        counter(tensor_input)
        str_1 = 'Input image resolution: ({}, {}, {}, {})'.format(_batch_size, *_input_shape)
        str_2 = 'Trainable model parameters: {}'.format(self.count_parameters())
        str_3 = 'Flops: {}'.format(flops_counter_.flops_to_string(counter.compute_average_flops_cost()))
        printf(str_1, _is_pause=False)
        printf(str_2, _is_pause=False)
        printf(str_3, _is_pause=False)
        printf('----------------------------------------------------------------', _is_pause=False)

        return '{}; {}; {}.'.format(str_1, str_2, str_3)

    def count_parameters(self) -> int:
        # Another option
        # return filter(lambda p: p.requires_grad, self.parameters())
        # return sum([np.prod(p.size()) for p in model_parameters])
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
