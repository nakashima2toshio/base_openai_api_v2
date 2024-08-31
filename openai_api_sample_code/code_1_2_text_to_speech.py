# Text to Speech
from openai import OpenAI
from pathlib import Path
# from a1_chat_completions import create_chat_completions
import pprint

from openai_api_sample_code.code_1_0_chat_completions import create_chat_completions

client = OpenAI()

model_text_to_speech = "tts-1"
model_audio_transcription = "whisper-1"


def create_audio_speech(input_text, speech_file_path):
    # (1) 入力テキストからオーディオを生成します。
    response = client.audio.speech.create(
        model=model_text_to_speech,
        voice="nova",
        input=input_text  # "The quick brown fox jumped over the lazy dog."
    )
    response.stream_to_file(speech_file_path)


def create_audio_transcriptions(speech_mp3, speech_file_path):
    # (2) デフォルト: 音声を入力言語に書き起こします。
    audio_file = open(speech_mp3, "rb")
    transcript = client.audio.transcriptions.create(
        model=model_audio_transcription,
        file=audio_file
    )
    return transcript


def create_audio_transcription_word_timestamps(speech_mp3):
    # (3) 単語のタイムスタンプ: 音声を入力言語に書き起こします。
    # タイムスタンプの粒度を使用するには、response_format or verbose_jsonを設定する必要があります
    audio_file = open(speech_mp3, "rb")
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model=model_audio_transcription,
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    return transcript


def create_audio_transcriptions_segment_timestamps(speech_mp3):
    # (4) セグメント・タイムスタンプ
    # タイムスタンプの粒度を使用するには、response_format or verbose_jsonを設定する必要があります
    audio_file = open(speech_mp3, "rb")
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model=model_audio_transcription,
        response_format="verbose_json",
        timestamp_granularities=["segment"]
    )
    return transcript


def create_audio_transcription(speech_file_mp3):
    # (5) 音声を英語に翻訳します。
    audio_file = open(speech_file_mp3, "rb")
    transcript = client.audio.translations.create(
        model=model_audio_transcription,
        file=audio_file
    )
    return transcript

def main():
    # (1) 入力テキストからオーディオを生成します。
    input_text = "Hello, my name is Wolfgang and I come from Germany. Where are you heading today?"
    speech_file_path = './speech_file.mp3'
    create_audio_speech(input_text, speech_file_path)

    # (1) def create_chat_completions(system_content, user_content)
    system_content = "あなたは有能な翻訳者のアシスタントです。"
    user_content = "プロのソフトウェア開発者向けに、OpenAiのAPIの概要を説明しなさい。"
    chat_messages = create_chat_completions(system_content, user_content)
    print(chat_messages.content)

if __name__ == '__main__':
    main()

"""
response =
{
  "text": "Hello, my name is Wolfgang and I come from Germany. Where are you heading today?"
}
"""
pass
""" 単語のタイムスタンプ
{
  "task": "transcribe",
  "language": "english",
  "duration": 8.470000267028809,
  "text": "The beach was a popular spot on a hot summer day. People were swimming in the ocean, building sandcastles, and playing beach volleyball.",
  "words": [
    {
      "word": "The",
      "start": 0.0,
      "end": 0.23999999463558197
    },
    ...
    {
      "word": "volleyball",
      "start": 7.400000095367432,
      "end": 7.900000095367432
    }
  ]
}

"""
pass
""" セグメントのタイムスタンプ
response = 
{
  "task": "transcribe",
  "language": "english",
  "duration": 8.470000267028809,
  "text": "The beach was a popular spot on a hot summer day. People were swimming in the ocean, building sandcastles, and playing beach volleyball.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.319999933242798,
      "text": " The beach was a popular spot on a hot summer day.",
      "tokens": [
        50364, 440, 7534, 390, 257, 3743, 4008, 322, 257, 2368, 4266, 786, 13, 50530
      ],
      "temperature": 0.0,
      "avg_logprob": -0.2860786020755768,
      "compression_ratio": 1.2363636493682861,
      "no_speech_prob": 0.00985979475080967
    },
    ...
  ]
}

"""

# verbose json
"""
{
  "task": "transcribe",
  "language": "english",
  "duration": 8.470000267028809,
  "text": "The beach was a popular spot on a hot summer day. People were swimming in the ocean, building sandcastles, and playing beach volleyball.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.319999933242798,
      "text": " The beach was a popular spot on a hot summer day.",
      "tokens": [
        50364, 440, 7534, 390, 257, 3743, 4008, 322, 257, 2368, 4266, 786, 13, 50530
      ],
      "temperature": 0.0,
      "avg_logprob": -0.2860786020755768,
      "compression_ratio": 1.2363636493682861,
      "no_speech_prob": 0.00985979475080967
    },
    ...
  ]
}

"""