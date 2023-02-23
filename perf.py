import logging as log
from argparse import ArgumentParser
import torch
from model import LwPosr
from ptflops import get_model_complexity_info
log.basicConfig(level=log.INFO)


# Count Model Parameters
def get_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Model Size in MB
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def get_model_complexity(model, input_shape=(3, 450, 450)):
    # Calculate Model MACC's required per Layer
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=False, verbose=True)
    return macs, params


def main(args):
    # Random Input Tensor
    x = torch.rand(1, 3, 450, 450)
    print(f"Input Tensor Shape: {x.shape}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = LwPosr(input_ch=x.shape[1], output_ch=3)
    model = model.to(device)

    if args.get_params:
        log.info(f"Model Parameters: {get_model_parameters(model)}")
    
    if args.get_size:
        if args.half:
            # FP16
            log.info(f"Model Size (FP16): {get_model_size(model.half()) :.3f} MB\n")
        else:
            # FP32
            log.info(f"Model Size (FP32): {get_model_size(model) :.3f} MB\n")
    
    if args.get_complexity:
        macs, params = get_model_complexity(model)
        log.info(f"Computational complexity (MAC's): {macs :<8}\n")
        log.info(f"Number of parameters: {params :<8}\n")
    
    else:
        return NotImplementedError
    
    return


def build_parser():
    parser = ArgumentParser(prog="LwPosr Model Performance Evaluation")
    parser.add_argument("-i", "--fpath_model", required=True, type=str,
                        help="Required. Path to saved model state dictionary.")
    parser.add_argument("-p", "--get_params", action='store_true',
                        help="Get total number of Model Trainable Parameters")
    parser.add_argument("-s", "--get_size", action='store_true',
                        help="Get Model Size in MB")
    parser.add_argument("-half" , "--half", action='store_true', 
                        help='Use FP16 half-precision')
    parser.add_argument("-c", "--get_complexity", action='store_true',
                        help="Get Model Layerwise Complexity as MAC's & FLOPS")

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
