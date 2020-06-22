# Snake
Inspired by ["Neural Networks Fail to Learn Periodic Functions and How to Fix It"](https://arxiv.org/abs/2006.08195).

This is a PyTorch implementation of the `snake` activation function from the paper - or at least I _think_ it is, no affiliation with the authors, use at your own risk, etc., etc.

A few variations of the function are discussed in the paper, this package implements:

 <img src="https://render.githubusercontent.com/render/math?math=x%2B\frac{1}{a}sin^{2}(ax)">

## Installation
From the main repo folder, run `python setup.py install`

## Usage
Fairly easy really `from snake.activations import Snake`.  The `Snake` constructor [(code here)](snake/activations.py) has an optional **learnable** parameter alpha which defaults to 1.  The authors of the paper find values between 5 and 50 work quite well for "known-periodic" data, while for better results with non-periodic data, you should choose a small value like 0.2.  The constructor also takes an `alpha_learnable` parameter which defaults to `True`, so that you can disable "learnability" if your experiments so require.

## Sample code
There's a notebook, still quite rough - [example.ipynb](example.ipynb).  


## Acknowledgements
This code probably wouldn't have gotten written if it hadn't been for [Alexandra Deis](https://towardsdatascience.com/@astakhova.aleksandra) and [her excellent article](https://towardsdatascience.com/extending-pytorch-with-custom-activation-functions-2d8b065ef2fa) .
