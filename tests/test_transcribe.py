import functools
import inspect
import os

from types import SimpleNamespace

import numpy as np
import pytest

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
from faster_whisper.vad import collect_chunks


def test_supported_languages():
    model = WhisperModel("tiny.en")
    assert model.supported_languages == ["en"]


def test_transcribe(jfk_path):
    model = WhisperModel("tiny")
    segments, info = model.transcribe(jfk_path, word_timestamps=True)
    assert info.all_language_probs is not None

    assert info.language == "en"
    assert info.language_probability > 0.9
    assert info.duration == 11

    # Get top language info from all results, which should match the
    # already existing metadata
    top_lang, top_lang_score = info.all_language_probs[0]
    assert info.language == top_lang
    assert abs(info.language_probability - top_lang_score) < 1e-16

    segments = list(segments)

    assert len(segments) == 1

    segment = segments[0]

    assert segment.text == (
        " And so my fellow Americans, ask not what your country can do for you, "
        "ask what you can do for your country."
    )

    assert segment.text == "".join(word.word for word in segment.words)
    assert segment.start == segment.words[0].start
    assert segment.end == segment.words[-1].end
    batched_model = BatchedInferencePipeline(model=model)
    result, info = batched_model.transcribe(
        jfk_path, word_timestamps=True, vad_filter=False
    )
    assert info.language == "en"
    assert info.language_probability > 0.7
    segments = []
    for segment in result:
        segments.append(
            {"start": segment.start, "end": segment.end, "text": segment.text}
        )

    assert len(segments) == 1
    assert segment.text == (
        " And so my fellow Americans ask not what your country can do for you, "
        "ask what you can do for your country."
    )


def test_batched_transcribe(physcisworks_path):
    model = WhisperModel("tiny")
    batched_model = BatchedInferencePipeline(model=model)
    result, info = batched_model.transcribe(physcisworks_path, batch_size=16)
    assert info.language == "en"
    assert info.language_probability > 0.7
    segments = []
    for segment in result:
        segments.append(
            {"start": segment.start, "end": segment.end, "text": segment.text}
        )
    # number of near 30 sec segments
    assert len(segments) == 6

    result, info = batched_model.transcribe(
        physcisworks_path,
        batch_size=16,
        without_timestamps=False,
        word_timestamps=True,
    )
    segments = []
    for segment in result:
        assert segment.words is not None
        segments.append(
            {"start": segment.start, "end": segment.end, "text": segment.text}
        )
    assert len(segments) > 7


def test_empty_audio():
    audio = np.asarray([], dtype="float32")
    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model=model)
    assert list(model.transcribe(audio)[0]) == []
    assert list(pipeline.transcribe(audio)[0]) == []
    model.detect_language(audio)


def test_prefix_with_timestamps(jfk_path):
    model = WhisperModel("tiny")
    segments, _ = model.transcribe(jfk_path, prefix="And so my fellow Americans")
    segments = list(segments)

    assert len(segments) == 1

    segment = segments[0]

    assert segment.text == (
        " And so my fellow Americans, ask not what your country can do for you, "
        "ask what you can do for your country."
    )

    assert segment.start == 0
    assert 10 < segment.end <= 11


def test_vad(jfk_path):
    model = WhisperModel("tiny")
    segments, info = model.transcribe(
        jfk_path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
    )
    segments = list(segments)

    assert len(segments) == 1
    segment = segments[0]

    assert segment.text == (
        " And so my fellow Americans ask not what your country can do for you, "
        "ask what you can do for your country."
    )

    assert 0 < segment.start < 1
    assert 10 < segment.end < 11

    assert info.vad_options.min_silence_duration_ms == 500
    assert info.vad_options.speech_pad_ms == 200


def test_stereo_diarization(data_dir):
    model = WhisperModel("tiny")

    audio_path = os.path.join(data_dir, "stereo_diarization.wav")
    left, right = decode_audio(audio_path, split_stereo=True)

    segments, _ = model.transcribe(left)
    transcription = "".join(segment.text for segment in segments).strip()
    assert transcription == (
        "He began a confused complaint against the wizard, "
        "who had vanished behind the curtain on the left."
    )

    segments, _ = model.transcribe(right)
    transcription = "".join(segment.text for segment in segments).strip()
    assert transcription == "The horizon seems extremely distant."


def test_multilingual_transcription(data_dir):
    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model)

    audio_path = os.path.join(data_dir, "multilingual.mp3")
    audio = decode_audio(audio_path)

    segments, info = model.transcribe(
        audio,
        multilingual=True,
        without_timestamps=True,
        condition_on_previous_text=False,
    )
    segments = list(segments)

    assert (
        segments[0].text
        == " Permission is hereby granted, free of charge, to any person obtaining a copy of the"
        " software and associated documentation files to deal in the software without restriction,"
        " including without limitation the rights to use, copy, modify, merge, publish, distribute"
        ", sublicence, and or cell copies of the software, and to permit persons to whom the "
        "software is furnished to do so, subject to the following conditions. The above copyright"
        " notice and this permission notice, shall be included in all copies or substantial "
        "portions of the software."
    )

    assert (
        segments[1].text
        == " Jedem, der dieses Software und die dazu gehöregen Dokumentationsdatein erhält, wird "
        "hiermit unengeltlich die Genehmigung erteilt, wird der Software und eingeschränkt zu "
        "verfahren. Dies umfasst insbesondere das Recht, die Software zu verwenden, zu "
        "vervielfältigen, zu modifizieren, zu Samenzofügen, zu veröffentlichen, zu verteilen, "
        "unterzulizenzieren und oder kopieren der Software zu verkaufen und diese Rechte "
        "unterfolgen den Bedingungen anderen zu übertragen."
    )

    segments, info = pipeline.transcribe(audio, multilingual=True)
    segments = list(segments)

    assert (
        segments[0].text
        == " Permission is hereby granted, free of charge, to any person obtaining a copy of the"
        " software and associated documentation files to deal in the software without restriction,"
        " including without limitation the rights to use, copy, modify, merge, publish, distribute"
        ", sublicence, and or cell copies of the software, and to permit persons to whom the "
        "software is furnished to do so, subject to the following conditions. The above copyright"
        " notice and this permission notice, shall be included in all copies or substantial "
        "portions of the software."
    )
    assert (
        "Dokumentationsdatein erhält, wird hiermit unengeltlich die Genehmigung erteilt,"
        " wird der Software und eingeschränkt zu verfahren. Dies umfasst insbesondere das Recht,"
        " die Software zu verwenden, zu vervielfältigen, zu modifizieren"
        in segments[1].text
    )


def test_hotwords(data_dir):
    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model)

    audio_path = os.path.join(data_dir, "hotwords.mp3")
    audio = decode_audio(audio_path)

    segments, info = model.transcribe(audio, hotwords="ComfyUI")
    segments = list(segments)

    assert "ComfyUI" in segments[0].text
    assert info.transcription_options.hotwords == "ComfyUI"

    segments, info = pipeline.transcribe(audio, hotwords="ComfyUI")
    segments = list(segments)

    assert "ComfyUI" in segments[0].text
    assert info.transcription_options.hotwords == "ComfyUI"


def test_transcribe_signature():
    model_transcribe_args = set(inspect.getargs(WhisperModel.transcribe.__code__).args)
    pipeline_transcribe_args = set(
        inspect.getargs(BatchedInferencePipeline.transcribe.__code__).args
    )
    pipeline_transcribe_args.remove("batch_size")

    assert model_transcribe_args == pipeline_transcribe_args


def test_monotonic_timestamps(physcisworks_path):
    model = WhisperModel("base")
    pipeline = BatchedInferencePipeline(model=model)

    segments, info = model.transcribe(physcisworks_path, word_timestamps=True)
    segments = list(segments)

    for i in range(len(segments) - 1):
        assert segments[i].start <= segments[i].end
        assert segments[i].end <= segments[i + 1].start
        for word in segments[i].words:
            assert word.start <= word.end
            assert word.end <= segments[i].end
    assert segments[-1].end <= info.duration

    segments, info = pipeline.transcribe(physcisworks_path, word_timestamps=True)
    segments = list(segments)

    for i in range(len(segments) - 1):
        assert segments[i].start <= segments[i].end
        assert segments[i].end <= segments[i + 1].start
        for word in segments[i].words:
            assert word.start <= word.end
            assert word.end <= segments[i].end
    assert segments[-1].end <= info.duration


def test_cliptimestamps_segments(jfk_path):
    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model=model)

    audio = decode_audio(jfk_path)
    audio = np.concatenate([audio, audio])
    clip_timestamps = [{"start": 0.0, "end": 11.0}, {"start": 11.0, "end": 22.0}]

    segments, info = pipeline.transcribe(audio, clip_timestamps=clip_timestamps)
    segments = list(segments)

    assert len(segments) == 2
    for segment, clip in zip(segments, clip_timestamps):
        assert segment.start == clip["start"]
        assert segment.end == clip["end"]
        assert segment.text == (
            " And so my fellow Americans ask not what your country can do for you, "
            "ask what you can do for your country."
        )


def test_cliptimestamps_timings(physcisworks_path):
    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model=model)

    audio = decode_audio(physcisworks_path)
    clip_timestamps = [{"start": 0.0, "end": 5.0}, {"start": 6.0, "end": 15.0}]
    transcripts = [
        " Now I want to return to the conservation of mechanical energy.",
        (
            " I have here a pendulum. I have an object that weighs 15 kilograms"
            " and I can lift it up one meter, which I have done now."
        ),
    ]
    segments, info = pipeline.transcribe(audio, clip_timestamps=clip_timestamps)
    segments = list(segments)

    assert len(segments) == 2
    for segment, clip, transcript in zip(segments, clip_timestamps, transcripts):
        assert clip["start"] == segment.start
        assert clip["end"] == segment.end
        assert segment.text == transcript


class _StopAfterVad(Exception):
    pass


def _run_batched_vad(monkeypatch, audio, vad_parameters):
    import faster_whisper.transcribe as transcribe_module

    captured = {}
    real_get_speech_timestamps = transcribe_module.get_speech_timestamps
    real_collect_chunks = transcribe_module.collect_chunks

    def spy_get_speech_timestamps(audio, vad_options=None, **kwargs):
        captured["vad_options"] = vad_options
        return real_get_speech_timestamps(audio, vad_options, **kwargs)

    def spy_collect_chunks(*args, **kwargs):
        captured["chunks"] = real_collect_chunks(*args, **kwargs)
        raise _StopAfterVad

    monkeypatch.setattr(
        transcribe_module, "get_speech_timestamps", spy_get_speech_timestamps
    )
    monkeypatch.setattr(transcribe_module, "collect_chunks", spy_collect_chunks)

    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model=model)
    try:
        pipeline.transcribe(audio, vad_parameters=vad_parameters)
    except _StopAfterVad:
        pass
    return captured


def test_batched_vad_options_instance_max_speech(
    physcisworks_path, monkeypatch, caplog
):
    from faster_whisper.vad import VadOptions

    audio = decode_audio(physcisworks_path)
    vad_options = VadOptions()
    with caplog.at_level("WARNING", logger="faster_whisper"):
        captured = _run_batched_vad(monkeypatch, audio, vad_options)

    assert "max_speech_duration_s is larger than chunk_length" in caplog.text
    assert captured["vad_options"].max_speech_duration_s == 30
    _, chunks_metadata = captured["chunks"]
    assert all(chunk["duration"] <= 30 for chunk in chunks_metadata)
    assert all(chunk["duration"] > 0 for chunk in chunks_metadata)

    # The caller's object must not be modified.
    assert vad_options == VadOptions()


def test_batched_vad_options_instance_smaller_max_speech_kept(
    jfk_path, monkeypatch, caplog
):
    from faster_whisper.vad import VadOptions

    audio = decode_audio(jfk_path)
    vad_options = VadOptions(max_speech_duration_s=5)
    with caplog.at_level("WARNING", logger="faster_whisper"):
        captured = _run_batched_vad(monkeypatch, audio, vad_options)

    assert "max_speech_duration_s" not in caplog.text
    assert captured["vad_options"] is vad_options
    assert captured["vad_options"].max_speech_duration_s == 5


def test_batched_vad_parameters_dict_not_mutated(jfk_path, monkeypatch):
    audio = decode_audio(jfk_path)
    vad_parameters = dict(max_speech_duration_s=10, min_silence_duration_ms=500)
    captured = _run_batched_vad(monkeypatch, audio, vad_parameters)

    assert captured["vad_options"].max_speech_duration_s == 30
    assert captured["vad_options"].min_silence_duration_ms == 500
    assert vad_parameters == dict(max_speech_duration_s=10, min_silence_duration_ms=500)


def test_collect_chunks_first_chunk_exceeds_max_duration():
    audio = np.arange(40000, dtype=np.float32)
    chunks = [{"start": 0, "end": 20000}, {"start": 25000, "end": 30000}]

    audio_chunks, chunks_metadata = collect_chunks(
        audio, chunks, sampling_rate=16000, max_duration=1
    )

    assert len(audio_chunks) == 2
    np.testing.assert_array_equal(audio_chunks[0], audio[0:20000])
    np.testing.assert_array_equal(audio_chunks[1], audio[25000:30000])
    assert chunks_metadata == [
        {"offset": 0.0, "duration": 1.25, "segments": [chunks[0]]},
        {"offset": 1.25, "duration": 0.3125, "segments": [chunks[1]]},
    ]


def test_collect_chunks():
    audio = np.arange(40000, dtype=np.float32)
    chunks = [
        {"start": 1000, "end": 7000},
        {"start": 8000, "end": 14000},
        {"start": 20000, "end": 26000},
        {"start": 30000, "end": 32000},
    ]

    audio_chunks, chunks_metadata = collect_chunks(
        audio, chunks, sampling_rate=16000, max_duration=1
    )

    assert len(audio_chunks) == 2
    np.testing.assert_array_equal(
        audio_chunks[0], np.concatenate([audio[1000:7000], audio[8000:14000]])
    )
    np.testing.assert_array_equal(
        audio_chunks[1], np.concatenate([audio[20000:26000], audio[30000:32000]])
    )
    assert chunks_metadata == [
        {"offset": 0.0, "duration": 0.75, "segments": chunks[0:2]},
        {"offset": 0.75, "duration": 0.5, "segments": chunks[2:4]},
    ]

    audio_chunks, chunks_metadata = collect_chunks(audio, chunks)

    assert len(audio_chunks) == 1
    np.testing.assert_array_equal(
        audio_chunks[0], np.concatenate([audio[c["start"] : c["end"]] for c in chunks])
    )
    assert chunks_metadata == [{"offset": 0.0, "duration": 1.25, "segments": chunks}]


class _FakeTimestampTokenizer:
    timestamp_begin = 50364

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens if t < self.timestamp_begin)


def _make_fake_split_model():
    model = SimpleNamespace(time_precision=0.02, input_stride=2, frames_per_second=50)
    model._split_segments_by_timestamps = functools.partial(
        WhisperModel._split_segments_by_timestamps, model
    )
    return model


def _ts(seconds):
    return _FakeTimestampTokenizer.timestamp_begin + round(seconds / 0.02)


def test_split_segments_by_timestamps_unfinished_tail():
    tokenizer = _FakeTimestampTokenizer()
    model = _make_fake_split_model()
    # <|0.00|> 1 2 <|2.00|><|2.40|> 3 4  (no closing timestamp for "3 4")
    tokens = [_ts(0.0), 1, 2, _ts(2.0), _ts(2.4), 3, 4]
    kwargs = dict(
        tokenizer=tokenizer,
        tokens=tokens,
        time_offset=10.0,
        segment_size=250,
        segment_duration=5.0,
        seek=0,
    )

    # Sequential path: the unfinished tail is dropped and re-decoded from seek.
    segments, seek, single_timestamp_ending = model._split_segments_by_timestamps(
        **kwargs
    )
    assert not single_timestamp_ending
    assert len(segments) == 1
    assert segments[0]["tokens"] == tokens[:4]
    assert seek == 200

    # Batched path: the unfinished tail is kept as the last segment.
    segments, seek, single_timestamp_ending = model._split_segments_by_timestamps(
        **kwargs, keep_unfinished_segment=True
    )
    assert not single_timestamp_ending
    assert seek == 200
    assert len(segments) == 2
    assert segments[0]["tokens"] == tokens[:4]
    assert segments[1]["tokens"] == tokens[4:]
    assert segments[1]["start"] == pytest.approx(12.4)
    assert segments[1]["end"] == pytest.approx(15.0)


def test_split_segments_by_timestamps_tail_without_text():
    tokenizer = _FakeTimestampTokenizer()
    model = _make_fake_split_model()
    # <|0.00|> 1 2 <|2.00|><|2.40|>  (tail contains only a timestamp)
    tokens = [_ts(0.0), 1, 2, _ts(2.0), _ts(2.4)]
    segments, _, _ = model._split_segments_by_timestamps(
        tokenizer=tokenizer,
        tokens=tokens,
        time_offset=0.0,
        segment_size=250,
        segment_duration=5.0,
        seek=0,
        keep_unfinished_segment=True,
    )
    assert [s["tokens"] for s in segments] == [tokens[:4]]


def test_batched_forward_keeps_unfinished_tail(monkeypatch):
    tokenizer = _FakeTimestampTokenizer()
    pipeline = BatchedInferencePipeline(_make_fake_split_model())
    tokens = [_ts(0.0), 1, 2, _ts(2.0), _ts(2.4), 3, 4]
    monkeypatch.setattr(
        pipeline,
        "generate_segment_batched",
        lambda *args: (
            None,
            [dict(tokens=tokens, avg_logprob=-0.1, no_speech_prob=0.0)],
        ),
    )
    options = SimpleNamespace(word_timestamps=False)
    chunks_metadata = [dict(offset=10.0, duration=5.0)]

    (segments,) = pipeline.forward(None, tokenizer, chunks_metadata, options)

    assert [s["text"] for s in segments] == ["1 2", "3 4"]
    assert segments[1]["start"] == pytest.approx(12.4)
    assert segments[1]["end"] == pytest.approx(15.0)


def test_batched_word_timestamps_interleaved_generators(jfk_path):
    model = WhisperModel("tiny")
    pipeline = BatchedInferencePipeline(model=model)

    jfk = decode_audio(jfk_path)
    silence = np.zeros(5 * 16000, dtype=jfk.dtype)
    inputs = [
        (np.concatenate([silence, jfk]), [{"start": 0.0, "end": 16.0}]),
        (
            np.concatenate([jfk, silence, jfk]),
            [{"start": 0.0, "end": 11.0}, {"start": 11.0, "end": 27.0}],
        ),
    ]

    def transcribe(audio, clip_timestamps):
        segments, _ = pipeline.transcribe(
            audio,
            clip_timestamps=clip_timestamps,
            batch_size=1,
            word_timestamps=True,
        )
        return segments

    def get_words(segments):
        return [(w.word, w.start, w.end) for s in segments for w in s.words]

    expected = [get_words(transcribe(*args)) for args in inputs]

    # Consume two generators of the same pipeline in an interleaved way.
    generator_a, generator_b = [transcribe(*args) for args in inputs]
    segments_b = [next(generator_b)]
    segments_a = [next(generator_a)]
    segments_b.extend(generator_b)
    segments_a.extend(generator_a)

    assert get_words(segments_a) == expected[0]
    assert get_words(segments_b) == expected[1]
