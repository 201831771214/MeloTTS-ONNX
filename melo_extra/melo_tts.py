from melo.api import TTS
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf

class MeloTTSWrapper(nn.Module):
    def __init__(self, melo_tts:TTS, is_dynamic:bool=False):
        super().__init__()
        self.melo_tts = melo_tts
        self.is_dynamic = is_dynamic
        self.hop_size = 512
    
    @torch.no_grad()
    def forward(self,
                x_tst:torch.Tensor,
                x_tst_lengths:torch.Tensor,
                speakers:torch.Tensor,
                tones:torch.Tensor,
                lang_ids:torch.Tensor,
                bert:torch.Tensor,
                ja_bert:torch.Tensor,
                sdp_ratio:torch.Tensor,
                noise_scale:torch.Tensor,
                noise_scale_w:torch.Tensor,
                speed:torch.Tensor) -> torch.Tensor:
        """_summary_

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
            deterministic (bool, optional): deterministic. Defaults to True.
        
        Returns:
            audio (torch.Tensor): audio data. shape: (1, audio_len)
        """
        if self.is_dynamic:
            result = self.melo_tts.model.forward_for_export_dynamic(
                x=x_tst,
                x_lengths=x_tst_lengths,
                sid=speakers,
                tone=tones,
                language=lang_ids,
                bert=bert,
                ja_bert=ja_bert,
                noise_scale=noise_scale,
                speed=speed,
                noise_scale_w=noise_scale_w,
                sdp_ratio=sdp_ratio,
            )
            audio_tensor = result[0]
            
            print(f"audio_tensor: {audio_tensor} ---- shape: {audio_tensor.shape}")
            
            sf.write("test.wav", audio_tensor.squeeze(0).squeeze(0).numpy(), 44100)
                    
            audio_tensor = audio_tensor.squeeze(0) # remove channel dim
            return audio_tensor
        else:
            result = self.melo_tts.model.forward_for_export_static(
                x=x_tst,
                x_lengths=x_tst_lengths,
                sid=speakers,
                tone=tones,
                language=lang_ids,
                bert=bert,
                ja_bert=ja_bert,
                noise_scale=noise_scale,
                speed=speed,
                noise_scale_w=noise_scale_w,
                sdp_ratio=sdp_ratio,
            )

            audio_tensor = result[0]   # [b, 1, T_audio_full]
            y_lengths    = result[4]   # [b]，mel 帧数（有效帧）

            # 计算有效音频采样点数
            # Generator 的上采样倍率 = hop_size
            # 有效音频长度 = y_lengths * hop_size
            valid_audio_len = (y_lengths[0] * self.hop_size).item()  # batch=1 取第0个
            valid_audio_len = int(valid_audio_len)

            print(f"mel frames: {y_lengths[0].item()}, valid audio samples: {valid_audio_len}")
            print(f"full audio shape: {audio_tensor.shape}")

            # 截取有效音段 [b, 1, valid_audio_len]
            audio_valid = audio_tensor[:, :, :valid_audio_len]

            print(f"valid audio shape: {audio_valid.shape}")

            sf.write(
                "test.wav",
                audio_valid.squeeze(0).squeeze(0).numpy(),
                44100,
            )

            audio_valid = audio_valid.squeeze(0)  # [1, valid_audio_len]
            return audio_valid