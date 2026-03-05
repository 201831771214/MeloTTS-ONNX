import onnx
from onnx import helper
import argparse

# 数据类型映射表
DATA_TYPE_MAP = {
    1: "float32",  # onnx.TensorProto.FLOAT
    2: "uint8",    # onnx.TensorProto.UINT8
    3: "int8",     # onnx.TensorProto.INT8
    4: "uint16",   # onnx.TensorProto.UINT16
    5: "int16",    # onnx.TensorProto.INT16
    6: "int32",    # onnx.TensorProto.INT32
    7: "int64",    # onnx.TensorProto.INT64
    8: "string",   # onnx.TensorProto.STRING
    9: "bool",     # onnx.TensorProto.BOOL
    10: "float16", # onnx.TensorProto.FLOAT16
    11: "float64", # onnx.TensorProto.DOUBLE
    12: "uint32",  # onnx.TensorProto.UINT32
    13: "uint64",  # onnx.TensorProto.UINT64
    14: "complex64",  # onnx.TensorProto.COMPLEX64
    15: "complex128", # onnx.TensorProto.COMPLEX128
}

def export_onnx_info(model_path, output_file="model.info"):
    """
    将ONNX模型信息导出到文本文件
    
    参数:
        model_path (str): ONNX模型文件路径
        output_file (str): 输出信息文件路径 (默认: model.info)
    """
    # 加载ONNX模型
    model = onnx.load(model_path)
    
    filename_with_ext = model_path.split('/')[-1]
    filename = filename_with_ext.split('.')[0]
    output_file = f"./infos/{filename}.info"
    
    # 打开文件准备写入
    with open(output_file, 'w', encoding='utf-8') as f:
        # 1. 写入模型基本信息
        f.write("=" * 60 + "\n")
        f.write(f"ONNX模型基本信息\n")
        f.write("=" * 60 + "\n")
        f.write(f"模型文件路径: {model_path}\n")
        f.write(f"ONNX版本: {model.ir_version}\n")
        f.write(f"生产者信息: {model.producer_name} {model.producer_version}\n")
        f.write(f"模型版本: {model.model_version}\n")
        f.write(f"描述: {model.doc_string}\n\n")
        
        # 2. 写入模型输入信息
        f.write("=" * 60 + "\n")
        f.write(f"模型输入信息 (共 {len(model.graph.input)} 个输入)\n")
        f.write("=" * 60 + "\n")
        for i, input in enumerate(model.graph.input):
            f.write(f"Input {i+1}: {input.name}\n")
            in_dtype = DATA_TYPE_MAP[input.type.tensor_type.elem_type]
            f.write(f"  数据类型: {in_dtype}\n")
            shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
            f.write(f"  形状: {shape}\n")
            if input.doc_string:
                f.write(f"  描述: {input.doc_string}\n")
            f.write("\n")
        
        # 3. 写入模型输出信息
        f.write("=" * 60 + "\n")
        f.write(f"模型输出信息 (共 {len(model.graph.output)} 个输出)\n")
        f.write("=" * 60 + "\n")
        for i, output in enumerate(model.graph.output):
            f.write(f"Output {i+1}: {output.name}\n")
            out_dtype = DATA_TYPE_MAP[output.type.tensor_type.elem_type]
            f.write(f"  数据类型: {out_dtype}\n")
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            f.write(f"  形状: {shape}\n")
            if output.doc_string:
                f.write(f"  描述: {output.doc_string}\n")
            f.write("\n")
        
        # 4. 写入模型权重信息
        f.write("=" * 60 + "\n")
        f.write(f"模型权重信息 (共 {len(model.graph.initializer)} 个参数)\n")
        f.write("=" * 60 + "\n")
        total_params = 0
        for initializer in model.graph.initializer:
            dims = initializer.dims
            param_count = 1
            for dim in dims:
                param_count *= dim
            total_params += param_count
            
            # 方法1：直接使用映射表获取类型名
            if initializer.data_type in DATA_TYPE_MAP:
                dtype = DATA_TYPE_MAP[initializer.data_type]
            else:
                try:
                    # 方法2：尝试使用helper函数
                    np_dtype = helper.tensor_dtype_to_np_dtype(initializer.data_type)
                    dtype = np_dtype.name if hasattr(np_dtype, 'name') else str(np_dtype)
                except:
                    # 方法3：完全手动
                    dtype = f"未知类型({initializer.data_type})"
            
            f.write(f"参数名: {initializer.name}\n")
            f.write(f"  数据类型: {dtype}\n")
            f.write(f"  形状: {dims}\n")
            f.write(f"  参数数量: {param_count}\n")
            f.write("\n")
            
        f.write(f"模型总参数量: {total_params}\n\n")
        
        # 5. 写入运算符信息
        f.write("=" * 60 + "\n")
        f.write(f"模型运算符信息 (共 {len(model.graph.node)} 个运算符)\n")
        f.write("=" * 60 + "\n")
        op_counts = {}
        for node in model.graph.node:
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        # 按运算符使用频率排序
        sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
        
        f.write(f"{'运算符类型':<15} | 使用次数\n")
        f.write("-" * 30 + "\n")
        for op_type, count in sorted_ops:
            f.write(f"{op_type:<15} | {count}\n")
        
        f.write("\n详细层信息:\n")
        f.write("-" * 60 + "\n")
        for i, node in enumerate(model.graph.node):
            f.write(f"层 {i+1}: {node.name} ({node.op_type})\n")
            f.write(f"  输入: {', '.join(node.input)}\n")
            f.write(f"  输出: {', '.join(node.output)}\n")
            f.write(f"  属性:\n")
            for attr in node.attribute:
                if attr.name in ['kernel_shape', 'strides', 'pads']:
                    value = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOAT:
                    value = attr.f
                else:
                    value = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else attr.s
                f.write(f"    - {attr.name}: {value}\n")
            f.write("\n")
        
        # 6. 写入元数据信息
        if model.metadata_props:
            f.write("\n" + "=" * 60 + "\n")
            f.write("模型元数据信息\n")
            f.write("=" * 60 + "\n")
            for key, value in model.metadata_props.items():
                f.write(f"{key}: {value}\n")
        
        # 7. 可选的额外信息
        f.write("\n" + "=" * 60 + "\n")
        f.write("额外信息\n")
        f.write("=" * 60 + "\n")
        f.write(f"导入版本: {', '.join([f'{opset.domain}-{opset.version}' for opset in model.opset_import])}\n")
        f.write(f"模型计算图名称: {model.graph.name}\n")
    
    print(f"模型信息已成功导出到 {output_file}")

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export ONNX model information')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Path to the ONNX model file')
    parser.add_argument('--output_file', '-o', type=str, default='model.info', help='Path to the output file')
    args = parser.parse_args()
    
    # onnx_model_path = "./models/TTS/outetts_onnx/model.onnx"  # 替换为你的ONNX模型路径
    
    onnx_model_path = args.model_path
    output_file = args.output_file
    export_onnx_info(onnx_model_path, output_file)