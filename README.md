# Reproducing Reasoning Emergence in Small LLMs via GRPO Post-Training

## How to load a model's weights
```
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

grpo_model = PeftModel.from_pretrained(base_model, "./directory/GRPO-lora")
tokenizer = AutoTokenizer.from_pretrained("./directory/GRPO-lora")```