import torch
import torch.nn as nn
import torchsummary
from abc import abstractmethod
from vujade.vujade_flops_counter import add_flops_counting_methods, flops_to_string


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        params = self.count_parameters()
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def _init_weights(self):
        print('The model name: {}.'.format(self.__class__.__name__))
        print('The weights of model are initialized.')

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_weights_kaming_normal(self, _scale=1.0, _a=0, _mode='fan_in', _nonlinearity='leaky_relu'):
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
        print('The model name: {}.'.format(self.__class__.__name__))
        print('The weights of model are initialized.')

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

    def summary(self, _input_shape, _batch_size=1, _device='cpu', _is_summary=True):
        print('{} Network summary.'.format(self.__class__.__name__))

        if _is_summary is True:
            torchsummary.summary(self, input_size=_input_shape, batch_size=_batch_size, device=_device)

        input = torch.randn([1, *_input_shape], dtype=torch.float).to(_device)
        counter = add_flops_counting_methods(self)
        counter.eval().start_flops_count()
        counter(input)
        print('Input image resolution:     ({:d}, {:d}, {:d}, {:d})'.format(_batch_size, _input_shape[0], _input_shape[1], _input_shape[2]))
        print('Trainable model parameters: {}'.format(self.count_parameters()))
        print('Flops:                      {}'.format(flops_to_string(counter.compute_average_flops_cost())))
        print('----------------------------------------------------------------')

    def count_parameters(self):
        # Another option
        # model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        # return sum([np.prod(p.size()) for p in model_parameters])
        return sum(p.numel() for p in self.parameters() if p.requires_grad)