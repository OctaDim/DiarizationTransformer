def get_speaker_types():
    speaker_types = {"operator": ["SPEAKER_00"],
                     "applicant": ["SPEAKER_01", "SPEAKER_02"]}
    return speaker_types


def calc_total_dur(segments):
    return max(segment["end"] for segment in segments)


def calc_speaker_dur(all_segments, speaker_categories):
    operator_duration = 0
    applicant_duration = 0
    for cur_segment in all_segments:
        cur_speaker = cur_segment["speaker"]
        duration = cur_segment["end"] - cur_segment["start"]

        if cur_speaker in speaker_categories["operator"]:
            operator_duration += duration
        elif cur_speaker in speaker_categories["applicant"]:
            applicant_duration += duration
    return operator_duration, applicant_duration


def calc_oper_appl_ratio(segments, speaker_categories):
    operator_dur, applicant_dur = calc_speaker_dur(segments, speaker_categories)
    if applicant_dur > 0:
        return round(operator_dur / applicant_dur, 3)
    else:
        return float('inf')


def count_interruptions(segments, speaker_categories):
    sorted_segments = sorted(segments, key=lambda x: x['start'])

    operator_interruptions = 0
    applicant_interruptions = 0
    current_speaker = None
    current_end = 0

    for segment in sorted_segments:
        speaker = segment['speaker']
        start = segment['start']

        # Check overlaps
        if current_speaker is not None and start < current_end:
            current_is_operator = current_speaker in speaker_categories['operator']
            new_is_operator = speaker in speaker_categories['operator']

            if new_is_operator and not current_is_operator:
                operator_interruptions += 1
            elif not new_is_operator and current_is_operator:
                applicant_interruptions += 1

        # Renew current speaker
        current_speaker = speaker
        current_end = segment['end']

    return operator_interruptions, applicant_interruptions


def calc_overlap_dur(segments, speaker_categories):
    events = []
    for segment in segments:
        events.append(('start', segment['start'], segment['speaker']))
        events.append(('end', segment['end'], segment['speaker']))

    events.sort(key=lambda x: x[1])

    active_speakers = set()
    prev_time = 0
    overlap_duration = 0

    for event_type, time, speaker in events:
        if active_speakers and time > prev_time:
            time_segment = time - prev_time
            has_operator = any(s in speaker_categories['operator'] for s in active_speakers)
            has_applicant = any(s in speaker_categories['applicant'] for s in active_speakers)

            if has_operator and has_applicant:
                overlap_duration += time_segment

        if event_type == 'start':
            active_speakers.add(speaker)
        else:
            active_speakers.discard(speaker)
        prev_time = time
    return overlap_duration


def calc_silence(segments, speaker_categories):
    total_dur = calc_total_dur(segments)
    operator_dur, applicant_dur = calc_speaker_dur(segments, speaker_categories)
    overlap_dur = calc_overlap_dur(segments, speaker_categories)

    total_speech_dur = operator_dur + applicant_dur - overlap_dur
    silence_dur = total_dur - total_speech_dur
    if total_dur > 0:
        return round((silence_dur / total_dur) * 100, 2)
    else:
        return 0


def calc_overlap(segments, speaker_categories):
    total_dur = calc_total_dur(segments)
    overlap_dur = calc_overlap_dur(segments, speaker_categories)
    if total_dur > 0:
        return round((overlap_dur / total_dur) * 100, 2)
    else:
        return 0


def complete_analyze(diarization_timings):
    segments = diarization_timings["diarization"]
    speaker_categories = get_speaker_types()

    oper_appl_ratio = calc_oper_appl_ratio(segments, speaker_categories)
    oper_intrpts, appl_intrpts = count_interruptions(segments, speaker_categories)
    overlap_percentage = calc_overlap(segments, speaker_categories)
    silence_percentage = calc_silence(segments, speaker_categories)

    operator_dur, applicant_dur = calc_speaker_dur(segments, speaker_categories)
    total_dur = calc_total_dur(segments)
    overlap_dur = calc_overlap_dur(segments, speaker_categories)

    return {
        "oper_appl_ratio": oper_appl_ratio,
        "appl_intrpts": appl_intrpts,
        "oper_intrpts": oper_intrpts,
        "overlap_percentage": overlap_percentage,
        "silence_percentage": silence_percentage,
        "operator_dur": round(operator_dur, 2),
        "applicant_dur": round(applicant_dur, 2),
        "total_dur": round(total_dur, 2),
        "overlap_dur": round(overlap_dur, 2)
    }


def demonstrate_parameters(diarization_timings):
    segments = diarization_timings["diarization"]
    speaker_categories = get_speaker_types()

    ratio = calc_oper_appl_ratio(segments, speaker_categories)
    print(f"Отношение оператор/заявитель: {ratio}")

    oper_intrpts, appl_intrpts = count_interruptions(segments, speaker_categories)
    print(f"Перебивания оператора: {oper_intrpts}")
    print(f"Перебивания заявителя: {appl_intrpts}")

    overlap_pct = calc_overlap(segments, speaker_categories)
    print(f"Доля одновременной речи: {overlap_pct}%")

    silence_pct = calc_silence(segments, speaker_categories)
    print(f"Процент тишины: {silence_pct}%")

    op_duration, app_duration = calc_speaker_dur(segments, speaker_categories)
    total_duration = calc_total_dur(segments)
    print(f"Длительность речи оператора: {op_duration:.2f} сек")
    print(f"Длительность речи заявителя: {app_duration:.2f} сек")
    print(f"Общая длительность: {total_duration:.2f} сек\n\n")
