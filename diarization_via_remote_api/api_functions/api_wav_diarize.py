import time

import requests
from pydub import AudioSegment


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


def test_api(api_key: str):
    api_test = requests.get(
        url='https://api.pyannote.ai/v1/test',
        headers={'Authorization': f'Bearer {api_key}',
                 'Content-Type': 'application/json'})
    return api_test


def upload_audio_file(input_path: str,
                      object_key: str,
                      api_key: str):
    presigned_url = None
    try:
        response = requests.post(
            url='https://api.pyannote.ai/v1/media/input',
            headers={'Authorization': f'Bearer {api_key}',
                     'Content-Type': 'application/json'},
            json={'url': f'media://{object_key}'})
        response.raise_for_status()
        presigned_url = response.json()['url']

        with open(input_path, 'rb') as file:
            file_data = file.read()
        response = requests.put(
            url=presigned_url,
            data=file_data,
            headers={'Content-Type': 'application/octet-stream'})
        response.raise_for_status()
        print(f'File uploaded [OK]')
        # print(f'input_path: {input_path}')
        # print(f'presigned_url: {presigned_url}')
        return presigned_url
    except requests.exceptions.RequestException as error:
        print(f'Uploading file [ERROR]: \n'
              f'input_path: {input_path}\n'
              f'presigned_url: {presigned_url}\n')
        raise error


def diarize_wav_by_object_key(object_key: str,
                              api_key: str):
    try:
        response = requests.post(
            url="https://api.pyannote.ai/v1/diarize",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"url": f"media://{object_key}",
                  "numSpeakers": 2})
        response.raise_for_status()
        response_json = response.json()
        print(f"response_json: {response}")
        print(f"response_json['jobId']: {response_json["jobId"]}")
        print(f"response_json['status']: {response_json["status"]}\n")
        # return response_json.json()
        return response_json["jobId"]
    except requests.exceptions.RequestException as error:
        print(f"Creating diarization job [ERROR]: "
              f"error: {error}\n"
              f"object_key: {object_key}\n"
              f"api_key: {api_key}\n")
        raise error


def get_result_by_job_id(api_key: str,
                         job_id: str):
    while True:
        response = requests.get(
            url=f"https://api.pyannote.ai/v1/jobs/{job_id}",
            headers={"Authorization": f"Bearer {api_key}"})
        if response.status_code != 200:
            print(f"Get job result by job id [ERROR]:\n"
                  f"response.status_code: {response.status_code}\n"
                  f"response.text: {response.text}\n")
            break
        response_json = response.json()
        response_status = response_json["status"]
        response_output = response_json["output"]
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"response_status: {response_status}")
        print(f"response_json: {response_json}")
        print(f"response_output: {response_output}")
        if response_status == "created":
            print(f"CURRENT JOB STATUS: '{response_status}' ==> waiting...\n")
            time.sleep(5)
            continue
        elif response_status in ["failed", "canceled"]:
            print(f"CURRENT JOB STATUS: '{response_status}' ==> broken\n")
            break
        elif response_status == "succeeded":
            print(f"CURRENT JOB STATUS: '{response_status}' ==> succeeded\n")
            return response_output
