import torch
from torch import sin, square, addcdiv
from torch.autograd import Function
from torch.nn import Parameter, Module
from torch.distributions.exponential import Exponential


class SnakeFunction(Function):
    """
    Autograd function implementing the serpentine-like sine-based periodic activation function.

    .. math::
         \text{Snake}_a := x + \frac{1}{a} \sin^2(ax)

    This function computes the forward and backward pass for the Snake activation, which helps in better
    extrapolating to unseen data, particularly when dealing with periodic functions.

    Attributes:
        ctx (torch.autograd.function._ContextMethodMixin): Context object used for saving and retrieving tensors.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Snake activation function.

        Args:
            x (torch.Tensor): Input tensor.
            a (torch.Tensor): Trainable parameter controlling the frequency of the sine function.

        Returns:
            torch.Tensor: Result of applying the Snake activation function to the input tensor.
        """
        ctx.save_for_backward(x, a)

        # Handle case where `a` is zero to avoid division by zero errors.
        return torch.where(a == 0, x, addcdiv(x, square(sin(a * x)), a))

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Backward pass for the Snake activation function.

        Args:
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The gradients of the loss with respect to `x` and `a`.
        """
        x, a = ctx.saved_tensors

        # Calculate the gradient of the input `x`
        sin2ax = sin(2 * a * x) if any(ctx.needs_input_grad) else None
        grad_x = grad_output * (1 + sin2ax) if ctx.needs_input_grad[0] else None

        # Calculate the gradient of the parameter `a`
        grad_a = (
            grad_output
            * torch.where(a == 0, square(x), sin2ax * x / a - square(sin(a * x) / a))
            if ctx.needs_input_grad[1]
            else None
        )

        return grad_x, grad_a


class Snake(Module):
    """
    Implementation of the Snake activation function as a torch module.

    .. math::
         \text{Snake}_a := x + \frac{1}{a} \sin^2(ax) = x - \frac{1}{2a}\cos(2ax) + \frac{1}{2a}

    This activation function is designed to better extrapolate unseen data, particularly periodic functions.

    Parameters:
        in_features (int or list): The shape or number of input features.
        a (float, optional): Initial value of the trainable parameter `a`, controlling the sine frequency. Defaults to None.
        trainable (bool, optional): If `True`, the parameter `a` will be trainable. Defaults to True.

    Examples:
        >>> snake_layer = Snake(256)
        >>> x = torch.randn(256)
        >>> x = snake_layer(x)

    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(
        self,
        in_features: int | list[int],
        a: float | None = None,
        trainable: bool = True,
    ):
        """
        Initialize the Snake activation layer.

        Args:
            in_features (int or list): Shape of the input, either a single integer or a list of integers indicating feature dimensions.
            a (float, optional): Initial value for the parameter `a`, which controls the sine frequency. If not provided, `a` will be initialized to a random value from an exponential distribution.
            trainable (bool, optional): If `True`, the parameter `a` will be trained during backpropagation.
        """
        super(Snake, self).__init__()
        self.in_features = (
            in_features if isinstance(in_features, list) else [in_features]
        )

        # Initialize `a`
        if a is not None:
            initial_a = torch.full(self.in_features, a)
        else:
            m = Exponential(torch.tensor([0.1]))
            initial_a = m.rsample(self.in_features).squeeze()

        if trainable:
            self.a = Parameter(initial_a)
        else:
            self.register_buffer("a", initial_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Snake activation layer.

        Args:
            x (torch.Tensor): Input tensor to the layer.

        Returns:
            torch.Tensor: Result of applying the Snake activation function.
        """
        return SnakeFunction.apply(x, self.a)
