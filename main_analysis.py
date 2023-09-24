import argparse
from model.mobileone import MobileOne
from vujade import vujade_torch as torch_
from vujade.vujade_debug import printd


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DNN analysis.')
    parser.add_argument('--mode', type=str, default='dev', help='Mode: dev; analysis; summary')
    args = parser.parse_args()

    return args


if __name__=='__main__':
    args = get_args()

    if args.mode not in {'dev', 'analysis', 'summary'}:
        raise ValueError('The mode is not supported.')

    input_res = (3, 32, 32)
    num_class = 10

    model_proposed = MobileOne(
        num_blocks_per_stage=[2, 8, 10, 1],
        num_classes=num_class,
        width_multipliers=(0.75, 1.0, 1.0, 2.0),
        inference_mode=False,
        use_se=False,
        num_conv_branches=4
    ).eval()
    model_proposed.eval()

    model = model_proposed

    if args.mode == 'dev':
        torch_.DNNComplexity(_model_cpu=model, _input_res=input_res).develop()
    elif args.mode == 'analysis':
        torch_.DNNComplexity(_model_cpu=model, _input_res=input_res).show()
    elif args.mode == 'summary':
        torch_.DNNComplexity(_model_cpu=model, _input_res=input_res).summary()
    else:
        raise ValueError('The mode is not supported.')
