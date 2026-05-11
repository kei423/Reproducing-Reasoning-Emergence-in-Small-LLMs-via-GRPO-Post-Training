# Reproducing Reasoning Emergence in Small LLMs via GRPO Post-Training


## Repository Organization


- `GRPO-lora`: Directory stores the weights of the GRPO model
- `SFT-lora`: Directory stores the weights of the SFT warmed-up model
- `CS272_Final_Project.ipynb`: Colab notebook containing all the training and evaluation code for the project
- `CS272_QuickStart_Test.ipynb`: Colab notebook showcasing an example for loading a model's weights and an example of prompting


## Dependencies


- `pip install torch transformers peft trl datasets bitsandbytes accelerate -q`
- `!pip install --upgrade --force-reinstall --no-cache-dir torchao -q`


## Requirements
- Linux or Google Colab recommended (run on Colab for best compatibility)
- Windows not supported (bitsandbytes dependency)
- Python 3.10+
- CUDA-capable GPU (A100 recommended)


## How to load a model's weights
Example below for GRPO model:
```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


model_name = "Qwen/Qwen2.5-1.5B-Instruct"


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)


grpo_model = PeftModel.from_pretrained(base_model, "./GRPO-lora")
tokenizer = AutoTokenizer.from_pretrained("./GRPO-lora")
grpo_model.eval()
```

