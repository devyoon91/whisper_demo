import torch

# pip install torch

if torch.cuda.is_available() is False:
    print("CUDA is impossible")
else:
    print("현재 디바이스 :{}".format(torch.cuda.current_device()))
    print("디바이스 갯수 :{}".format(torch.cuda.device_count()))

    for idx in range(0, torch.cuda.device_count()):
        print("디바이스 :{}".format(torch.cuda.device(idx)))
        print("디바이스 이름 :{}".format(torch.cuda.get_device_name(idx)))