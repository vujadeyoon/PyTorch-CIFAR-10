import argparse
import torchvision
from model.backbone import resnet
from model.model_8 import FaceParser
# from model.others.EAGRNet.networks.EAGR import EAGRNet
# from model.others.ibugRTN.ibug.face_parsing import SegmentationModel as ibgSegmentationModel
from vujade import vujade_dnn as dnn_
from vujade.vujade_debug import printf


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DNN analysis.')
    parser.add_argument('--mode', type=str, default='dev', help='Mode: dev; analysis; summary')
    args = parser.parse_args()

    return args

# (PyTorch) sjyoon1671@MATE:/DATA/sjyoon1671/Research/PyTorch-FaceParsing$ python3 main_analysis.py --mode dev
# Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth" to /home/sjyoon1671/.cache/torch/hub/checkpoints/efficientnet-b7-dcc49843.pth

if __name__=='__main__':
    args = get_args()

    if args.mode not in {'dev', 'analysis', 'summary'}:
        raise ValueError('The mode is not supported.')

    input_res = (3, 512, 512)
    num_classes = 11

    # model_eagr = EAGRNet(num_classes=num_classes).eval()
    # model_rtn = ibgSegmentationModel(encoder='rtnet50', decoder='fcn', num_classes=num_classes)
    # model_proposed = FaceParser(num_classes=num_classes, backbone_name='efficientnet_b0', backbone_pretrained=True).eval()
    # model_proposed = FaceParser(num_classes=num_classes, backbone_name='resnet50', backbone_pretrained=True).eval()
    model_proposed = FaceParser(num_classes=num_classes, backbone_name='resnet18', backbone_pretrained=True).eval()
    model_proposed.eval()

    model = model_proposed

    if args.mode == 'dev':
        dnn_.DNNComplexity(_model_cpu=model, _input_res=input_res).develop()
    elif args.mode == 'analysis':
        dnn_.DNNComplexity(_model_cpu=model, _input_res=input_res).show()
    elif args.mode == 'summary':
        dnn_.DNNComplexity(_model_cpu=model, _input_res=input_res).summary()
    else:
        raise ValueError('The mode is not supported.')


    # EAGRNet
    # Trainable params.: 66.72 M
    # Macs:              269.51 GMac
    # Flops:             539.02 GFlop
    #
    #
    # RTNet-50
    # Trainable params.: 27.33 M
    # Macs:              115.48GMac
    # Flops:             230.96 GFlop
