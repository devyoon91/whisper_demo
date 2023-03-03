import time

import whisper

# print available model names
print(whisper.available_models())

# param : model, device[cpu, cuda(default)]
# model : tiny -> tiny.en -> base -> ...
# 한국어는 미디엄 이상되야 품질이 나온다고함
model = whisper.load_model("small", "cpu")

# FP16 is not supported on CPU; using FP32 instead
# ../samples/event_horizon.mp3 -> 윤하 - 사건의 지평선
audio = "../samples/event_horizon.mp3"
language = "ko"

# options : beam_size ?
# options : best_of ?
# options : fp16 ?
# options = dict(language=language, beam_size=5, best_of=5, fp16=False)
options = dict(language=language, fp16=False)
transcribe_options = dict(task="transcribe", **options)

start = time.time()
result = model.transcribe(audio, **transcribe_options)
end = time.time()

# print the recognized text
print("language : ", result['language'])
print("text : ", result['text'])
print(f"executed time {end - start:.5f} sec")
print("===============================================================================================================")

# print segments
for segment in result['segments']:
    print("segments : ", segment)
