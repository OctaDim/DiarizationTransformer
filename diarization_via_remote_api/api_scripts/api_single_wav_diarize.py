# ######################################################################
# ########################## DEXP API OK ###############################
# ######################################################################

import os.path
import uuid

from diarization_statistics_functions.statistics_calculations import complete_analyze, demonstrate_parameters
from diarization_via_remote_api.api_functions.api_wav_diarize import (
    convert_audio_ogg_to_wav, diarize_wav_by_object_key, get_result_by_job_id, test_api,
    upload_audio_file)


ogg_full_file_path = r"C:\Users\dexp\Projects\DiarizationTransformer\audio_samples\ogg\4.ogg"
wav_full_file_path = r"C:\Users\dexp\Projects\DiarizationTransformer\audio_samples\wav\4.wav"
# ogg_full_file_path = "/home/octadim/PycharmProjects/DiarizationTransformer/audio_samples/ogg/1.ogg"
# wav_full_file_path = "/home/octadim/PycharmProjects/DiarizationTransformer/audio_samples/wav/1.wav"
# wav_full_file_path = os.path.normpath(wav_full_file_path)
ogg_full_file_path = os.path.normpath(ogg_full_file_path)

api_key = "sk_0d466286b6784130a2d669e1a1ab481f"
object_key = f"{uuid.uuid4()}.wav"

convert_audio_ogg_to_wav(
    from_ogg_file_path=ogg_full_file_path,
    to_wav_file_path=wav_full_file_path,
    start_time_msec=7000,
    end_time_msec=60000)

print(f"API TEST WITHOUT AUTHENTICATION: {test_api(api_key)}\n")

presigned_url = upload_audio_file(
    # input_path=ogg_full_file_path,
    input_path=wav_full_file_path,
    object_key=object_key,
    api_key=api_key)
print(f"FUNC RETURN: presigned_url: {presigned_url}\n")

job_id = diarize_wav_by_object_key(
    object_key=object_key,
    api_key=api_key)
print(f"FUNC RETURN: job_id: {job_id}\n")

response_output = get_result_by_job_id(
    api_key=api_key,
    job_id=job_id)
print(f"FUNC RETURN: response_output: {response_output}\n")
for cur_num, cur_timing in enumerate(response_output["diarization"]):
    print(f"{cur_num}\t speaker: {cur_timing["speaker"]}\t"
          f"{cur_timing["start"]}\t - {cur_timing["end"]}")
print("\n")

demonstrate_parameters(diarization_timings=response_output)

results = complete_analyze(diarization_timings=response_output)
for key, value in results.items():
    print(f"{key}: {value}")
