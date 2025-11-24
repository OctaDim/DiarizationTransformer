# pip install pyannote.audio
# ######################################################################
# ################### DEXP LOCAL WINDOWS ERROR #########################
# ################ 'AudioDecoder' is not defined #######################
# ######################################################################


import os
from os import PathLike
from typing import Literal, Union

import huggingface_hub
import torch
import torchaudio
import transformers
from huggingface_hub import login
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.pipeline.typing import PipelineOutput
from pydub import AudioSegment


transformers.logging.set_verbosity_debug()  # Transformers logging switching on
huggingface_hub.logging.set_verbosity_debug()  # HF logging switching on

hf_token = "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"
api_token = "sk_0644cc93474640fbbd9479d4b8dc269c"
login(token=hf_token)


# Just normalizing full file path
def normalize_file_path(file_path: str):
    full_file_name_path = file_path.replace("\\", "/")  # for Windows, for Linux not necessary
    full_file_name_path = os.path.normpath(path=full_file_name_path)
    print(f"full_file_name_path: {full_file_name_path}")
    return full_file_name_path


# Converting .ogg to .wav (mono, 16000)
def convert_audio_ogg_to_wav(
        from_ogg_file_path,
        to_wav_file_path,
        start_time_msec=None,
        end_time_msec=None):
    """Convert .ogg to .wav
    :param from_ogg_file_path: str or PathLike: full .ogg file path including filename with extension
    :param to_wav_file_path: str or PathLike: full .wav file path including filename with extension
    :param start_time_msec: int: e.g. 7000. Converted file starts from 7 sec
    :param end_time_msec: int: e.g. 60000. Converted file ends in 1 min
    :return: None (save .wav file into to_wav_file_path path)"""
    audio = AudioSegment.from_ogg(from_ogg_file_path)
    audio = audio[start_time_msec:end_time_msec]
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(to_wav_file_path, format="wav")
    print(f"Converted [OK]: \n"
          f"from_ogg_file_path: {from_ogg_file_path}\n"
          f"to_wav_file_path: {to_wav_file_path}\n")
    return to_wav_file_path


# diarization wav audio via diarizing ML Model and return output
def diarize_wav_audio(
        full_wav_file_path: Union[str, PathLike],
        model_checkpoint: str = "pyannote/speaker-diarization-3.1",
        device_type: Union[str, Literal["cpu", "cuda"]] = "cuda",
        num_speakers: int = 2,
        use_torchaudio_load: bool = True
) -> PipelineOutput:
    try:
        pipeline = Pipeline.from_pretrained(
            checkpoint=model_checkpoint,
            token=hf_token, )
        print(f"Pipeline.from_pretrained(): {pipeline}")
        pipeline = pipeline.to(torch.device(device_type))
        print(f"pipeline.to(torch.device(device_type): {pipeline}\n")

        if not use_torchaudio_load:
            with ProgressHook() as progress_hook:
                model_output = pipeline(file=full_wav_file_path,
                                        num_speakers=num_speakers,
                                        hook=progress_hook)
        else:
            waveform, sample_rate = torchaudio.load(
                uri=full_wav_file_path)
            model_output = pipeline({"waveform": waveform,
                                     "sample_rate": sample_rate})

        print(f"diarizing process [OK]:\n"
              f"model_output: {model_output}\n")
        return model_output
    except Exception as error:
        print(f"diarizing process [ERROR]: error: {error}\n"
              f"full_wav_file_path: {full_wav_file_path}\n"
              f"model_checkpoint: {model_checkpoint}\n"
              f"device_type: {device_type}\n"
              f"use_torchaudio_load: {use_torchaudio_load}\n")


# Absolute path to .ogg file, which will be converted to .wav file.
ogg_full_file_path = r"C:\Users\dexp\Projects\DiarizationTransformer\audio_samples\ogg\3.ogg"

# Absolute path to .wav file, which will be created via converting and which will be transferred to the model
wav_full_file_path = r"C:\Users\dexp\Projects\DiarizationTransformer\audio_samples\wav\3.wav"

# Just normalizing files paths
ogg_full_file_path = normalize_file_path(ogg_full_file_path)
wav_full_file_path = normalize_file_path(wav_full_file_path)

# Converting .ogg => .wav
convert_audio_ogg_to_wav(
    from_ogg_file_path=ogg_full_file_path,
    to_wav_file_path=wav_full_file_path,
    start_time_msec=7000,
    end_time_msec=60000)

# ######################################################################
# ############# pyannote model "speaker-diarization-3.1" ###############
# ######################################################################
print(f"\n\n{'#' * 75}")
print("PYANNOTE LOCAL MODEL: 'speaker-diarization-3.1'")
print(f"{'#' * 75}")

# One of the 2 models to try diarization with ML Model "speaker-diarization-3.1"
model_checkpoint = "pyannote/speaker-diarization-3.1"

# Auto-determining the type: CPU or CUDA
device_name = "cuda" if torch.cuda.is_available() else "cpu"

# Function with calling model pipline
output_model_diarization_3_1 = diarize_wav_audio(
    full_wav_file_path=wav_full_file_path,
    model_checkpoint=model_checkpoint,
    device_type=device_name,  # "cuda" or "cpu"
    num_speakers=2,
    use_torchaudio_load=False)
# If output exists model is working perfect
print(f"output_model_diarization_3_1: {output_model_diarization_3_1}\n")

######################################################################
########## pyannote model "speaker-diarization-community-1" ##########
######################################################################
print(f"\n\n{'#' * 75}")
print("PYANNOTE LOCAL MODEL: 'speaker-diarization-community-1'")
print(f"{'#' * 75}")

# One of the 2 models to try diarization with ML Model "speaker-diarization-community-1"
model_checkpoint = "pyannote/speaker-diarization-community-1"

# Auto-determining the type: CPU or CUDA
device_name = "cuda" if torch.cuda.is_available() else "cpu"

output_model_community_1 = diarize_wav_audio(
    full_wav_file_path=wav_full_file_path,
    model_checkpoint=model_checkpoint,
    device_type=device_name,  # "cuda" or "cpu"
    num_speakers=2,
    use_torchaudio_load=False)
print(f"output_model_community_1: {output_model_community_1}\n")
