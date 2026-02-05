# check_env.py
import torch
import transformers
import accelerate
import bitsandbytes
import datasets
import peft
import trl
import sentencepiece

print("✅ torch version:", torch.__version__)
print("✅ transformers version:", transformers.__version__)
print("✅ accelerate version:", accelerate.__version__)
print("✅ bitsandbytes version:", bitsandbytes.__version__)
print("✅ datasets version:", datasets.__version__)
print("✅ peft version:", peft.__version__)
print("✅ trl version:", trl.__version__)

print("\n모든 핵심 패키지 import 성공 ✅")
