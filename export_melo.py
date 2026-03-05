"""Export MeloTTS to ONNX Model.

    Args:
        x_tst (torch.Tensor): text token ids. shape: (batch_size, seq_len)
        x_tst_lengths (torch.Tensor): text token ids lengths. shape: (seq_len,)
        speakers (torch.Tensor): speaker ids. shape: (speaker_id,)
        tones (torch.Tensor): tone ids. shape: (batch_size, seq_len)
        lang_ids (torch.Tensor): language ids. shape: (batch_size, seq_len)
        bert (torch.Tensor): bert. shape: (batch_size, feats_dim, seq_len)
        ja_bert (torch.Tensor): ja_bert. shape: (batch_size, filter_channels, seq_len)
        sdp_ratio (torch.Tensor): sdp_ratio. shape: (ratio,)
        noise_scale (torch.Tensor): noise_scale. shape: (scale,)
        noise_scale_w (torch.Tensor): noise_scale_w. shape: (scale_w,)
        speed (torch.Tensor): speed. shape: (speed,)
    
    Returns:
        audio (torch.Tensor): audio data. shape: (1, audio_len)
"""

from melo_extra.melo_tts import MeloTTSWrapper
from melo.attentions import MultiHeadAttention
from melo.api import TTS
from melo.utils import get_text_for_tts_infer

import torch

import os
import onnx
from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
import sys

import traceback
import argparse
import time

import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("./logs/export_melotts.log", mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

EXTERNAL_DATA_THRESHOLD = 4 * 1024 * 1024 * 1024 # 4GB


def build_melo_dummy_input(
    melo_tts:TTS,
    test_text:str="我最近在学习machine learning。",
    sdp_ratio=0.2, 
    noise_scale=0.667, 
    noise_scale_w=0.8, 
    speed=1.0):
    """Build dummy input for MeloTTS.

    Args:
        batch_size (int, optional): batch size. Defaults to 1.
        seq_len (int, optional): sequence length. Defaults to 100.
        sdp_ratio (float, optional): sdp ratio. Defaults to 0.2.
        noise_scale (float, optional): noise scale. Defaults to 0.667.
        noise_scale_w (float, optional): noise scale w. Defaults to 0.8.
        speed (float, optional): speed. Defaults to 1.0.

    Returns:
        dict: dummy input for MeloTTS.
    """
    bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(test_text, 'ZH_MIX_EN', melo_tts.hps, melo_tts.device, melo_tts.symbol_to_id)
    
    
    x_tst = phones.to(torch.int32).unsqueeze(0)
    x_tst_lengths = torch.Tensor([phones.size(0)]).to(torch.int32)
    speakers = torch.Tensor([1]).to(torch.int32)
    tones = tones.to(torch.int32).unsqueeze(0)
    lang_ids = lang_ids.to(torch.int32).unsqueeze(0)
    
    bert = bert.to(torch.float32).unsqueeze(0)
    ja_bert = ja_bert.to(torch.float32).unsqueeze(0)
    
    ratio = torch.Tensor([sdp_ratio]).to(torch.float32)
    scale = torch.Tensor([noise_scale]).to(torch.float32)
    scale_w = torch.Tensor([noise_scale_w]).to(torch.float32)
    a_speed = torch.Tensor([speed]).to(torch.float32)
    
    return (x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert, ratio, scale, scale_w, a_speed)

def export_melotts(
    melo_tts:TTS,
    output_path:str,
    dummy_input,
    opset_version:int=14,
    is_dynamic:bool=True
):
    melotts_wrapper = MeloTTSWrapper(melo_tts, is_dynamic).eval()
    
    
    input_names = ["x_tst", "x_tst_lengths", "speakers", "tones", "lang_ids", "bert", "ja_bert", "sdp_ratio", "noise_scale", "noise_scale_w", "speed"]
    output_names = ["audio_data"]
    
    if is_dynamic:
        dynamic_axes = {
            "x_tst": {0: "batch_size", 1: "sequence_length"},
            "x_tst_lengths": {0: "sequence_length"},
            "speakers": {0: "speaker_id"},
            "tones": {0: "batch_size", 1: "sequence_length"},
            "lang_ids": {0: "batch_size", 1: "sequence_length"},
            "bert": {0: "batch_size", 2: "sequence_length"},
            "ja_bert": {0: "batch_size", 2: "sequence_length"},
            "sdp_ratio": {0: "ratio"},
            "noise_scale": {0: "scale"},
            "noise_scale_w": {0: "scale_w"},
            "speed": {0: "speed"},
        }
        saved_name = f"melotts_{opset_version}_dynamic.onnx"
    else:
        dynamic_axes = {}
        saved_name = f"melotts_{opset_version}_static.onnx"
        
    try:
        os.makedirs(os.path.join(output_path, "melotts_onnx"), exist_ok=True)
        saved_root = os.path.join(output_path, "melotts_onnx")
        saved_path = os.path.join(saved_root, saved_name)
        
        
        if not is_dynamic:
            # 遍历完整模型的所有子模块
            count = 0
            for name, m in melo_tts.model.named_modules():
                if isinstance(m, MultiHeadAttention) and m.window_size is not None:
                    m._build_export_embeddings()
                    m.export_mode = True
                    count += 1

            print(f"\n共设置 {count} 个 MultiHeadAttention 层")

            # 验证没有遗漏
            missed = []
            for name, m in melo_tts.model.named_modules():
                if isinstance(m, MultiHeadAttention) and m.window_size is not None:
                    if not m.export_mode or m.emb_rel_k_export is None:
                        missed.append(name)

            if missed:
                print(f"以下层未设置成功: {missed}")
            else:
                print("所有层均已设置")
        
        traced_model = torch.jit.trace(melotts_wrapper, dummy_input, check_trace=True)
        
        torch.onnx.export(
            model=traced_model,
            args=dummy_input,
            f=saved_path,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            # external_data=True,
            do_constant_folding=True,
            export_modules_as_functions=False
        )

        # combine external data into main onnx file for easier loading in some runtimes (optional, can keep as external if preferred)
        # model_proto = onnx.load(saved_path)
        # load_external_data_for_model(model_proto, saved_root)
        # convert_model_to_external_data(
        #     model_proto,
        #     all_tensors_to_one_file=True,
        #     convert_attribute=True
        # )
        # onnx.save(model_proto, os.path.join(output_path, f"melotts_{opset_version}.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, size_threshold=EXTERNAL_DATA_THRESHOLD)
        logger.info(f"Melo TTS exported successfully to {os.path.join(output_path, saved_name)}")
    except Exception as e:
        logger.error(f"Export Melo TTS failed: {e}")
        logger.error(traceback.format_exc())
        raise e
    
default_ckpt_path = "./models/MeloTTS-Chinese/checkpoint.pth"
default_cfg_path = "./models/MeloTTS-Chinese/config.json"
default_output_path = "./models/"

default_test_txt = "我们正式推出大语言模型，旨在应对复杂系统工程和长周期智能体任务。扩展规模仍然是提升通用人工智能智能效率的最重要方式之一。"

if __name__ == "__main__":
    msg_info = f"Export Melo TTS model to {default_output_path} by default."
    usg_info = """
    Usage:
        python export_melotts.py [-m MODEL_PATH] [-o OUTPUT_PATH] [--opset OPSET] [-id] [-b BATCH_SIZE] [-s SEQUENCE_LENGTH]
    """

    parser = argparse.ArgumentParser(usage=usg_info, description=msg_info)
    parser.add_argument("-m", "--ckpt_path", type=str, default=default_ckpt_path, help="Path to the Melo TTS checkpoint.")
    parser.add_argument("-c", "--cfg_path", type=str, default=default_cfg_path, help="Path to the Melo TTS config file.")
    parser.add_argument("-o", "--output_path", type=str, default=default_output_path, help="Path to save the exported model.")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version to use.")
    parser.add_argument("-id", "--is_dynamic", action="store_true", help="Whether to export the model with dynamic axes.")
    parser.add_argument("-t", "--test_txt", type=str, default=default_test_txt, help="Test text for dummy input.")
    parser.add_argument("-sr", "--sdp_ratio", type=float, default=0.5, help="SDP ratio for dummy input.")     
    parser.add_argument("-ns", "--noise_scale", type=float, default=0.667, help="Noise scale for dummy input.")     
    parser.add_argument("-nsw", "--noise_scale_w", type=float, default=0.8, help="Noise scale w for dummy input.")     
    parser.add_argument("-sp", "--speed", type=float, default=1.0, help="Speed for dummy input.")     
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    logger.info(f"Export Moss TTS model to {args.output_path} by default.")
    start_time = time.perf_counter()
    
    melo_tts = TTS(
        language='ZH',
        device="cpu",
        use_hf=False,
        config_path=args.cfg_path,
        ckpt_path=args.ckpt_path,
    )
    
    dummy_input = build_melo_dummy_input(
        melo_tts=melo_tts,
        test_text=args.test_txt,
        sdp_ratio=args.sdp_ratio,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        speed=args.speed,
    )

    export_melotts(
        melo_tts=melo_tts,
        output_path=args.output_path,
        opset_version=args.opset,
        dummy_input=dummy_input,
        is_dynamic=args.is_dynamic,
    )
    logger.info(f"Export Melo TTS model done in {time.perf_counter() - start_time:.2f} seconds.")