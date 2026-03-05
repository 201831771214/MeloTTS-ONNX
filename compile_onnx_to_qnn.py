import qai_hub as hub
import logging
import os
import sys
import time
import argparse

from melo_extra.inireader import IniReader

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("./logs/compile_melo_onnx.log", mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

def compile_onnx_to_qnn(onnx_model_path: str, output_path: str, chip_name:str="QCS8550", job_name:str = "compile_melotts", input_specs:dict=None):
    """
    编译 ONNX 模型为 QNN 模型

    :param onnx_model_path: ONNX 模型路径
    :param qnn_model_path: QNN 模型路径
    """
    
    # 获取Qualcomm Device
    qual_devices = hub.get_devices()
    selected_device = None
    
    for q_dev in qual_devices:
        logger.info(f"Qualcomm Device: {q_dev}")
        if chip_name.lower() in q_dev.name.lower():
            selected_device = q_dev
            break

    if selected_device is None:
        logger.error(f"Qualcomm Device {chip_name} not found.")
        raise ValueError(f"Qualcomm Device {chip_name} not found.")
    
    if input_specs is None or not isinstance(input_specs, dict):
        logger.error("input_specs must be a dict. and must contain all onnx model input specs.")
        raise ValueError("input_specs must be a dict. and must contain all onnx model input specs.")
    
    # 构建Compile Job
    compile_job = hub.submit_compile_job(
        model=onnx_model_path,
        device=selected_device,
        name=job_name,
        input_specs=input_specs,
        options="--target_runtime precompiled_qnn_onnx --output_names audio_data"
    )
    
    tar_attributes = getattr(selected_device, 'attributes', 'N/A')
    
    logger.info(f"compile_job: {compile_job}")
    logger.info(f"编译任务已提交，任务ID: {getattr(compile_job, 'job_id', 'N/A')}")
    logger.info(f"编译任务模型ID: {getattr(compile_job, 'model_id', 'N/A')}")
    logger.info(f"目标设备: {selected_device}")
    for attr in tar_attributes:
        logger.info(f"  - {attr}")
        
    logger.info("等待编译完成...")
    
    # 等待编译完成
    compile_job.wait()
    
    job_status = compile_job.get_status()
    logger.info(f"编译任务状态: {job_status}")
    
    compile_log_path = os.path.join("./compile_logs/", getattr(compile_job, 'job_id', 'N/A'))
    os.makedirs(compile_log_path, exist_ok=True)
    if job_status.code == 'FAILED':
        error_msg = job_status.message
        logger.error(f"编译失败: {error_msg}")
        logger.error(f"任务状态: {job_status}")
        compile_job.download_results(compile_log_path)
        return None
    else:
        success_msg = job_status.message
        compile_data = compile_job.download_results(compile_log_path)
        logger.info(f"编译成功: {success_msg}")
        logger.info(f"编译数据：{compile_data}")
    
    logger.info("编译成功完成！")
    
    # 构建Profile Job
    
    compile_target_model = compile_job.get_target_model()
    
    logger.info("运行性能分析...")
    profile_job = hub.submit_profile_job(
        model=compile_target_model,
        device=selected_device,
        name=job_name,
    )
    
    profile_job.wait()
    
    # 检查性能分析状态
    profile_status = profile_job.get_status()
    profile_log_path = os.path.join("./profile_logs/", getattr(profile_job, 'job_id', 'N/A'))
    os.makedirs(profile_log_path, exist_ok=True)
    logger.info(f"性能分析任务状态: {profile_status}")
    if profile_status.code == 'FAILED':
        error_msg = profile_status.message
        logger.error(f"性能分析失败: {error_msg}")
        logger.error(f"任务状态: {profile_status.code}")
        profile_job.download_results(profile_log_path)
    else:
        profile_data = profile_job.download_results(profile_log_path)
        logger.info("性能分析完成")
        success_msg = profile_status.message
        logger.info(f"性能分析数据：{profile_data}")
        logger.info(f"性能分析：{success_msg}")
    
    output_qnn_path = os.path.join(output_path, f"{job_name}")
    os.makedirs(output_qnn_path, exist_ok=True)
    profiled_model = profile_job.model.download(output_qnn_path)
    logger.info(f"所有任务均已完成，模型保存在: {output_qnn_path}")
        
default_onnx_path = "./models/melotts_onnx/melotts_14_static.onnx"
default_output_qnn_path = "./models/melotts_qnn/"
default_cfg_path = "./compile_configs/cfg.ini"
default_target_device = "QCS8550"

if __name__ == "__main__":
    msg_info="Compile ONNX format model to QNN format"
    usg_info="""
    Usage:
        python compile_onnx_to_qnn.py -m <ONNX_MODEL_PATH> -o <QNN_MODEL_PATH> -jn <JOB_NAME> -td <TARGET_DEVICE> -cp <CFG_PATH>
    """
    
    parser = argparse.ArgumentParser(usage=usg_info, description=msg_info)
    parser.add_argument("-m", "--model_path", type=str, default=default_onnx_path, help="ONNX model path")
    parser.add_argument("-o", "--output_path", type=str, default=default_output_qnn_path, help="QNN model path")
    parser.add_argument("-jn", "--job_name", type=str, default="melotts_14", help="job name")
    parser.add_argument("-td", "--target_device", type=str, default=default_target_device, help="target device")
    parser.add_argument("-cp", "--cfg_path", type=str, default=default_cfg_path, help="compile config path")
    args = parser.parse_args()
    
    ini_reader = IniReader(args.cfg_path)
    
    cfg = ini_reader.GetConfig()
    model_input_specs = cfg["input_specs"]
    
    try:
        start_time = time.perf_counter()
        compile_onnx_to_qnn(
            onnx_model_path=args.model_path, 
            output_path=args.output_path, 
            chip_name=args.target_device, 
            job_name=args.job_name, 
            input_specs=model_input_specs)
        logger.info(f"编译耗时: {(time.perf_counter() - start_time):.4f}/s")
    except Exception as e:
        logger.error(f"编译失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)