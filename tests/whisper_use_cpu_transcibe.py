import time

import whisper

# param : model, device[cpu, cuda(default)]
# model : tiny -> tiny.en -> base -> ...
# 한국어는 미디엄 이상되야 품질이 나온다고함
model = whisper.load_model("base", "cpu")

# FP16 is not supported on CPU; using FP32 instead
# ../samples/event_horizon.mp3 -> 윤하 - 사건의 지평선
start = time.time()
result = whisper.transcribe(model=model, audio="../samples/event_horizon.mp3", fp16=False)
end = time.time()

# print the recognized text
print("language : ", result['language'])
print("segments : ", result['segments'])
print("text : ", result['text'])
print(f"{end - start:.5f} sec")
