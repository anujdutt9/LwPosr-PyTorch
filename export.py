# Export Trained Model to ONNX/CoreML/TF/TFLite/TF.js
import os
import logging as log
from argparse import ArgumentParser
import torch
import numpy as np
from model import LwPosr
import onnx
from onnxsim import simplify
log.basicConfig(level=log.INFO)


def export_model(args):
    """ Function to convert the model from pytorch to specified format
    :param args: input args
    :return: Converted Model
    """

    model = LwPosr(input_ch=3, output_ch=3)
    model.load_state_dict(torch.load(args.fpath_model))
    model.eval()
    model.to('cpu')

    # Sample Input Tensor
    x = torch.randn(1, 3, 450, 450, requires_grad=False)
    out = torch.randn(1, 3)
    print("input_shape: ", x.shape)
    print("output_shape: ", out.shape)

    if str(args.output_format).lower() == "onnx":
        # Convert to ONNX format
        onnx_model_path = os.path.join(args.dirpath_out, f"LwPosr.onnx")
        torch.onnx.export(model, x, onnx_model_path,
                          export_params=True, opset_version=12, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        # Check ONNX Model
        onnx_model = onnx.load(os.path.join(args.dirpath_out, f"LwPosr.onnx"))
        onnx.checker.check_model(onnx_model)
        # Print a Human readable representation of the graph
        onnx.helper.printable_graph(onnx_model.graph)
        log.info(f"ONNX model saved at {os.path.join(args.dirpath_out, f'LwPosr.onnx')}\n")

        if args.simplify_onnx:
            model_simplified, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
            # Save Simplified Model
            onnx.save(onnx_model, os.path.join(args.dirpath_out, f'LowPosr_Simplified.onnx'))
            log.info(f"Simplified ONNX model saved at {os.path.join(args.dirpath_out, f'LowPosr_Simplified.onnx')}\n")

    elif str(args.output_format).lower() == "coreml":
        import coremltools as ct

        traced_model = torch.jit.trace(model, x)
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="input", shape=x.shape)],
        )
        # Save model
        coreml_model.save(os.path.join(args.dirpath_out, f"LwPosr.mlmodel"))
        log.info(f"CoreML model saved at {os.path.join(args.dirpath_out, f'LwPosr.mlmodel')}\n")

    elif str(args.output_format).lower() == "tf" or str(args.output_format).lower() == "tflite" or str(args.output_format).lower() == "tfjs":
        import tensorflow as tf
        from onnx_tf.backend import prepare

        # Convert to ONNX format
        onnx_model_path = os.path.join(args.dirpath_out, f"LwPosr.onnx")
        # Model output in SavedModel format
        tf_model_path = os.path.join(args.dirpath_out, f"LwPosr")
        torch.onnx.export(model, x, os.path.join(args.dirpath_out, f"LwPosr.onnx"),
                          export_params=True, opset_version=12, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        onnx_model = onnx.load(onnx_model_path)
        tf_rep = prepare(onnx_model)
        # Export TF Model
        tf_rep.export_graph(tf_model_path)
        # Test Converted Model
        model = tf.saved_model.load(tf_model_path)
        model.trainable = False
        input_tensor = tf.random.uniform([1, 3, 450, 450])
        out = model(**{'input': input_tensor})
        log.info(f"TensorFlow Model Output Shape: {out['output'].shape}")
        log.info(f"TensorFlow model saved at {tf_model_path}\n")

        # Convert TF to TFLite
        if str(args.output_format).lower() == "tflite":
            # TFLite model path
            tflite_model_path = os.path.join(args.dirpath_out, f"LwPosr.tflite")
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
                tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
            ]
            tflite_model = converter.convert()
            # Save the model
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)

            # Test TFLite Model
            # Load the TFLite model and allocate tensors
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            interpreter.allocate_tensors()
            # Get input and output tensors
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            # Test the model on random input data
            input_shape = input_details[0]['shape']
            input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            # get_tensor(): returns a copy of the tensor data
            output_data = interpreter.get_tensor(output_details[0]['index'])
            log.info(f"TFLite Output Shape: {output_data.shape}")
            log.info(f"TensorFlow Lite model saved at {tflite_model_path}")

        elif str(args.output_format).lower() == "tfjs":
            import subprocess

            tfjs_model_path = os.path.join(args.dirpath_out, f"LwPosr")
            out = subprocess.run("tensorflowjs_converter "
                                 "--input_format=tf_saved_model "
                                 "--output_node_names='output' "
                                 "--saved_model_tags=serve "
                                 f"{tf_model_path} "
                                 f"{tfjs_model_path} ", shell=True)
            print(out)
            log.info(f"TensorFlow JS model saved at {tfjs_model_path}")

    else:
        return NotImplementedError


def build_parser():
    parser = ArgumentParser(prog="LwPosr Export")
    parser.add_argument("-i", "--fpath_model", required=True, type=str,
                        help="Required. Path to saved model state dictionary.")
    parser.add_argument("-o", "--dirpath_out", required=True, type=str,
                        help="Required. Path to save exported model.")
    parser.add_argument("-simplify", "--simplify_onnx", action='store_true',
                        help="Optional. Simplify ONNX Model.")
    parser.add_argument("-f", "--output_format", required=True, type=str,
                        help="Required. Model export format. ex. ONNX, CoreML, TF, TFLite, TFjs")

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    export_model(args)