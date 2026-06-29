import qai_hub as qai
import os
import logging
from qai_hub import JobStatus

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("./logs/compile.log", mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

model_path = "./models/melotts_onnx/melotts_16_static_fp32.onnx"
output_path = "./outputs/"

compile_log_root = "./logs/compile/"
profile_log_root = "./logs/profile/"

class DeviceNotFoundError(Exception):
    pass

class FrameWorkNotFoundError(Exception):
    pass

class CompileJobFailedError(Exception):
    pass

def get_all_supported_devices():
    device_list = qai.get_devices()
    for dev in device_list:
        logger.info(f"{dev.name}")

def compile_model(model_path:str, chip_name="QCS8550 (Proxy)", framework_version="2.46", job_name="kokoro_onnx_qnn", output_path=output_path):
    os.makedirs(output_path, exist_ok=True)
    
    device_name = [dev.name for dev in qai.get_devices()]
    device_attr = [dev.attributes for dev in qai.get_devices()]
    
    if chip_name in device_name:
        logger.info(f"Use Chipset: {chip_name}")
        idx = device_name.index(chip_name)
        attr_dict = dict(item.split(':', 1) for item in device_attr[idx])
        logger.info(f"Chipset Attrs:\n {attr_dict}")
        device = qai.get_devices()[idx]
        logger.info(f"Qualcomm Dev: {device}")
    else:
        raise DeviceNotFoundError(
            f"Device {chip_name} not found. Supported Devices:\n" +
            "\n".join(f"  - {dev_n}" for dev_n in device_name)
        )
    
    frmaework_version_list = [framework.api_version for framework in qai.get_frameworks()]
    
    if framework_version in frmaework_version_list:
        logger.info(f"Use Framework: {framework_version}")
        idx = frmaework_version_list.index(framework_version)
        framework = qai.get_frameworks()[idx]
        logger.info(f"Framework: {framework}")
    else:
        raise FrameWorkNotFoundError(
            f"Framework {framework_version} not found. Supported Frameworks:\n" +
            "\n".join(f"  - {fw_v}" for fw_v in frmaework_version_list)
        )
    
    input_specs = {
        "x_tst": ((1, 512), "int32"),
        "x_tst_lengths": ((1, ), "int32"),
        "speakers": ((1, ), "int32"),
        "tones": ((1, 512), "int32"),
        "lang_ids": ((1, 512), "int32"),
        "bert": ((1, 1024, 512), "float32"),
        "ja_bert": ((1, 768, 512), "float32"),
        "sdp_ratio": ((1, ), "float32"),
        # "noise_scale_w": ((1, ), "float32"),
        "speed": ((1, ), "float32")
    }
    
    # Compile Model Job
    compile_job = qai.submit_compile_job(
        model=model_path,
        device=device,
        name=job_name,
        input_specs=input_specs,
        options=f"--qairt_version {framework_version} --target_runtime precompiled_qnn_onnx --output_names 'audio_data,y_lengths'"
    )
    
    # Wait for Job to Complete
    compile_status = compile_job.wait()
    
    compile_log_path = os.path.join(compile_log_root, compile_job.job_id)
    os.makedirs(compile_log_path, exist_ok=True)
    compile_job.download_job_logs(compile_log_path)
    
    if compile_status.state == JobStatus.State.FAILED:
        logger.error(f"Compile job failed: {compile_status.message}")
        raise CompileJobFailedError(f"Compile job failed: {compile_status.message}")
    
    target_model = compile_job.get_target_model()
    
    # Profile Model Job
    profile_job = qai.submit_profile_job(
        model=target_model,
        device=device,
        name=job_name,
    )
    
    # Workaround for qai_hub bug: ProfileJob._in_progress_states is missing RUNNING_INFERENCE
    profile_job._in_progress_states.append(JobStatus.State.RUNNING_INFERENCE)
    
    # Wait for Job to Complete
    profile_status = profile_job.wait()
    
    profile_log_path = os.path.join(profile_log_root, profile_job.job_id)
    os.makedirs(profile_log_path, exist_ok=True)
    profile_job.download_job_logs(profile_log_path)
    
    if profile_status.state == JobStatus.State.FAILED:
        logger.error(f"Profile job failed: {profile_status.message}")
        raise RuntimeError(f"Profile job failed: {profile_status.message}")
    
    target_model.download(output_path)
    logger.info(f"Download Target Model to {output_path}")
    logger.info(f"All Jobs Completed.")
        
if __name__ == "__main__":
    compile_model(model_path, chip_name="QCS8550 (Proxy)")