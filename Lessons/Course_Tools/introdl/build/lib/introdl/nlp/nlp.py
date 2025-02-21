import os
import gc
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import ipywidgets as widgets
import pyperclip
from IPython.display import display, clear_output, Markdown
from accelerate import cpu_offload
import contextlib

from transformers.utils import logging
logging.set_verbosity_error()

# Suppress logging warnings from PyTorch and Transformers
import logging
logging.getLogger("transformers.models.mistral.modeling_mistral").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Disable CUDA memory caching warning
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not compiled with flash attention.*")

# Global caches for Hugging Face models and tokenizers.
LOADED_MODELS = {}
LOADED_TOKENIZERS = {}

# --- Available Model Definitions ---

# Mapping short names to full model names.
SHORT_NAME_MAP = {
    "llama-3p1-8B": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "mistral-7B": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "llama-3p2-3B": "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    "gemini-flash-lite": "gemini-2.0-flash-lite-preview-02-05",
    "gemini-flash": "gemini-2.0-flash"
}

# Lists of API-based models.
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini"]
GEMINI_MODELS = ["gemini-2.0-flash-lite-preview-02-05", "gemini-2.0-flash"]

def llm_list_models():
    """
    Lists all available models grouped by category:
    - Short Name Models (Hugging Face & Gemini)
    - OpenAI Models
    - Gemini Models
    """
    models = {
        "Short Name Models": {k: v for k, v in SHORT_NAME_MAP.items()},
        "OpenAI Models": OPENAI_MODELS,
        "Gemini Models": GEMINI_MODELS
    }
    print("Available models:")
    for category, model_list in models.items():
        print(f"{category}:")
        if isinstance(model_list, dict):
            for short, full in model_list.items():
                print(f"  {short} => {full}")
        else:
            for model in model_list:
                print(f"  {model}")
    print("")  # Blank line after the list
    print("To use an OPENAI or GEMINI model, set the appropriate environment variable:")
    print("OPENAI_API_KEY or GEMINI_API_KEY")
    return models

def release_model():
    """Unloads the currently loaded Hugging Face model from memory and GPU."""
    global LOADED_MODELS, LOADED_TOKENIZERS
    if LOADED_MODELS:
        model_str = next(iter(LOADED_MODELS))
        print(f"🛑 Unloading model: {model_str} from GPU...")
        model = LOADED_MODELS[model_str]
        try:
            # If the model was loaded with accelerate, use safe offloading
            if hasattr(model, "hf_device_map"):
                #print("⚡ Using `accelerate.cpu_offload()` for safe unloading.")
                cpu_offload(model)
            else:
                print("⬇️ Moving model to CPU manually...")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="You shouldn't move a model that is dispatched using accelerate hooks")
                    model.to("cpu")

        except Exception as e:
            print(f"⚠️ Error moving model to CPU: {e}")
        del LOADED_MODELS[model_str]
        del LOADED_TOKENIZERS[model_str]
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
        print(f"✅ Model {model_str} has been fully unloaded.")
    else:
        print("❌ No model to unload.")

class ModelConfig:
    """
    Configuration object to store model, tokenizer, and API client settings.
    When using API-based models (e.g. OpenAI or Gemini), the model and tokenizer
    may be None.
    """
    def __init__(self, model_str, model=None, tokenizer=None, api_type=None,
                 cost_per_M_input=None, cost_per_M_output=None):
        self.model_str = model_str
        self.model = model
        self.tokenizer = tokenizer
        self.api_type = api_type  # e.g. "openai" or "gemini"
        self.client = None
        self.cost_per_M_input = cost_per_M_input
        self.cost_per_M_output = cost_per_M_output

        # Initialize API client if applicable.
        if api_type in ["openai", "gemini"]:
            api_key = os.getenv("GEMINI_API_KEY" if api_type == "gemini" else "OPENAI_API_KEY")
            if not api_key:
                raise ValueError(f"Missing {api_type.upper()} API key. Set the appropriate environment variable.")
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/" if api_type == "gemini" else None
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def unload(self):
        release_model()

    
def llm_configure(model_str, cost_per_M_input=None, cost_per_M_output=None):
    """
    Configures the model based on the provided model_str.
    For API-based models, returns a ModelConfig with the API client initialized.
    For Hugging Face models, manages the global cache (only one model loaded at a time).
    
    Parameters:
      - model_str: A model name string. Can be a short alias (e.g., "meta", "gemini-flash")
                   or a full Hugging Face model name.
      - cost_per_M_input, cost_per_M_output: Cost parameters for API-based models.
    
    Returns:
      A ModelConfig object.
    """
    model_str = model_str.strip()
    
    # Expand short names.
    if model_str in SHORT_NAME_MAP:
        model_str = SHORT_NAME_MAP[model_str]
    
    # Handle API-based models.
    if model_str in OPENAI_MODELS:
        return ModelConfig(model_str, api_type="openai",
                           cost_per_M_input=cost_per_M_input, cost_per_M_output=cost_per_M_output)
    if model_str in GEMINI_MODELS:
        return ModelConfig(model_str, api_type="gemini",
                           cost_per_M_input=cost_per_M_input, cost_per_M_output=cost_per_M_output)
    
    # For Hugging Face models: use the global cache (only one at a time).
    global LOADED_MODELS, LOADED_TOKENIZERS
    if model_str in LOADED_MODELS:
        # Model is already loaded; return silently.
        return ModelConfig(model_str, LOADED_MODELS[model_str], LOADED_TOKENIZERS[model_str])
    elif LOADED_MODELS:
        # A different model is loaded; release it.
        release_model()
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"🚀 Loading model: {model_str} (this may take a while)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_str,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        LOADED_MODELS[model_str] = model
        LOADED_TOKENIZERS[model_str] = tokenizer
        print(f"🟢 Model {model_str} loaded successfully.\n")
        return ModelConfig(model_str, model, tokenizer)
    except Exception as e:
        print(f"❌ Error loading model {model_str}: {str(e)}")
        return None


def clean_response(response, prompt=None):
    """
    Cleans the response by removing the input prompt, an 'assistant' label,
    and any extraneous blank lines.
    """
    if prompt:
        prompt_marker_pattern = re.escape(prompt) + r"\s*\.\s*assistant"
        match = re.search(prompt_marker_pattern, response)
        if match:
            response = response[match.end():].strip()
        else:
            match = re.search(re.escape(prompt), response)
            if match:
                response = response[match.end():].strip()
    response = re.sub(r"^\s*assistant\s*", "", response, flags=re.IGNORECASE).strip()
    response = "\n".join([line for line in response.split("\n") if line.strip()])
    return response

'''
def llm_prompt(model_config, prompts, max_new_tokens=200, temperature=0.7, 
               search_strategy="top_p", top_k=50, top_p=0.9, num_beams=1,
               estimate_cost=False, system_prompt="You are an AI assistant that provides brief answers."):
    """
    Generates a response from an LLM using a provided ModelConfig object.
    
    Parameters:
      - model_config: A preconfigured ModelConfig instance.
      - prompts: A single prompt (str) or a list of prompt strings.
      - max_new_tokens, temperature, search_strategy, top_k, top_p, num_beams: Generation parameters.
      - estimate_cost: If True and the cost parameters are set in ModelConfig, prints an estimated cost.
      - system_prompt: The system prompt to include in API-based chat calls.
    
    Returns:
      The generated response(s) as a string or a list of strings.
    """
    if model_config is None:
        return "❌ Error: Invalid model configuration. Please check the model name."
    
    is_batch = isinstance(prompts, list)
    responses = []
    num_input_tokens = 0.0
    num_output_tokens = 0.0

    # --- API-based models (OpenAI, Gemini) ---
    if model_config.api_type in ["openai", "gemini"]:
        try:
            messages = [{"role": "system", "content": system_prompt}]
            for prompt in ([prompts] if not is_batch else prompts):
                user_message = {"role": "user", "content": prompt}
                full_messages = messages + [user_message]
                response = model_config.client.chat.completions.create(
                    model=model_config.model_str,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens
                )
                response_text = response.choices[0].message.content.strip()
                responses.append(clean_response(response_text, prompt))
                if estimate_cost and model_config.cost_per_M_input is not None and model_config.cost_per_M_output is not None:
                    num_input_tokens += response.usage.prompt_tokens
                    num_output_tokens += response.usage.completion_tokens

            if estimate_cost:
                total_cost = ((num_input_tokens / 1_000_000) * model_config.cost_per_M_input) + \
                             ((num_output_tokens / 1_000_000) * model_config.cost_per_M_output)
                print(f"💰 Estimated Cost: ${total_cost:.6f} (Input: {num_input_tokens} tokens, Output: {num_output_tokens} tokens)")
            return responses if is_batch else responses[0]
        except Exception as e:
            return f"{model_config.api_type.capitalize()} API error: {str(e)}"

    # --- Local Hugging Face model generation ---
    tokenizer = model_config.tokenizer
    model = model_config.model

    if model is None or tokenizer is None:
        return "❌ Error: Model or tokenizer is not properly initialized."
    
    if hasattr(tokenizer, "apply_chat_template"):
        conversations = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
                         for p in (prompts if is_batch else [prompts])]
        input_ids = tokenizer.apply_chat_template(conversations, return_tensors="pt", padding=True, truncation=True).to(model.device)
    else:
        input_ids = tokenizer(prompts if is_batch else [prompts], return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": terminators,
        "repetition_penalty": 1.2,
        "num_beams": num_beams,
        "do_sample": temperature > 0,
        "temperature": temperature,
    }
    if search_strategy == "top_k":
        gen_kwargs.update({"do_sample": True, "top_k": top_k, "temperature": temperature})
    elif search_strategy == "top_p":
        gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})
    elif search_strategy == "contrastive":
        gen_kwargs.update({"do_sample": True, "penalty_alpha": 0.6, "top_k": 4, "temperature": temperature})

    with torch.no_grad():
        output = model.generate(input_ids, **gen_kwargs)
    responses = tokenizer.batch_decode(output, skip_special_tokens=True)
    responses = [clean_response(resp, prompt) for resp, prompt in zip(responses, prompts if is_batch else [prompts])]
    return responses if is_batch else responses[0]
'''

'''
def llm_generate(model_config, prompts, max_new_tokens=200, temperature=0.7, 
               search_strategy="top_p", top_k=50, top_p=0.9, num_beams=1,
               estimate_cost=False, system_prompt="You are an AI assistant that provides brief answers."):
    """
    Generates a response from an LLM using a provided ModelConfig object.
    
    Parameters:
      - model_config: A preconfigured ModelConfig instance.
      - prompts: A single prompt (str) or a list of prompt strings.
      - max_new_tokens, temperature, search_strategy, top_k, top_p, num_beams: Generation parameters.
      - estimate_cost: If True and the cost parameters are set in ModelConfig, prints an estimated cost.
      - system_prompt: The system prompt to include in API-based chat calls.
    
    Returns:
      The generated response(s) as a string or a list of strings.
    """
    if model_config is None:
        return "❌ Error: Invalid model configuration. Please check the model name."
    
    is_batch = isinstance(prompts, list)
    responses = []
    num_input_tokens = 0.0
    num_output_tokens = 0.0

    # --- API-based models (OpenAI, Gemini) ---
    if model_config.api_type in ["openai", "gemini"]:
        try:
            messages = [{"role": "system", "content": system_prompt}]
            for prompt in ([prompts] if not is_batch else prompts):
                user_message = {"role": "user", "content": prompt}
                full_messages = messages + [user_message]
                response = model_config.client.chat.completions.create(
                    model=model_config.model_str,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens
                )
                response_text = response.choices[0].message.content.strip()
                responses.append(clean_response(response_text, prompt))
                if estimate_cost and model_config.cost_per_M_input is not None and model_config.cost_per_M_output is not None:
                    num_input_tokens += response.usage.prompt_tokens
                    num_output_tokens += response.usage.completion_tokens

            if estimate_cost:
                total_cost = ((num_input_tokens / 1_000_000) * model_config.cost_per_M_input) + \
                             ((num_output_tokens / 1_000_000) * model_config.cost_per_M_output)
                print(f"💰 Estimated Cost: ${total_cost:.6f} (Input: {num_input_tokens} tokens, Output: {num_output_tokens} tokens)")
            return responses if is_batch else responses[0]
        except Exception as e:
            return f"{model_config.api_type.capitalize()} API error: {str(e)}"

    # --- Local Hugging Face model generation ---
    tokenizer = model_config.tokenizer
    model = model_config.model

    if model is None or tokenizer is None:
        return "❌ Error: Model or tokenizer is not properly initialized."
    
    # Check if a chat template is available; if not, fallback to normal tokenization.
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        conversations = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
                         for p in (prompts if is_batch else [prompts])]
        input_ids = tokenizer.apply_chat_template(conversations, return_tensors="pt", padding=True, truncation=True).to(model.device)
    else:
        input_ids = tokenizer(prompts if is_batch else [prompts], return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": terminators,
        "repetition_penalty": 1.2,
        "num_beams": num_beams,
        "do_sample": temperature > 0,
        "temperature": temperature,
    }
    if search_strategy == "top_k":
        gen_kwargs.update({"do_sample": True, "top_k": top_k, "temperature": temperature})
    elif search_strategy == "top_p":
        gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})
    elif search_strategy == "contrastive":
        gen_kwargs.update({"do_sample": True, "penalty_alpha": 0.6, "top_k": 4, "temperature": temperature})

    with torch.no_grad():
        output = model.generate(input_ids, **gen_kwargs)
    responses = tokenizer.batch_decode(output, skip_special_tokens=True)
    responses = [clean_response(resp, prompt) for resp, prompt in zip(responses, prompts if is_batch else [prompts])]
    return responses if is_batch else responses[0]


def llm_generate(model_config, prompts, max_new_tokens=200, temperature=0.7, 
                 search_strategy="top_p", top_k=50, top_p=0.9, num_beams=1,
                 estimate_cost=False, system_prompt="You are an AI assistant that provides brief answers."):
    """
    Generates a response from an LLM using a provided ModelConfig object.
    
    Parameters:
      - model_config: A preconfigured ModelConfig instance.
      - prompts: A single prompt (str) or a list of prompt strings.
      - max_new_tokens, temperature, search_strategy, top_k, top_p, num_beams: Generation parameters.
      - estimate_cost: If True and the cost parameters are set in ModelConfig, prints an estimated cost.
      - system_prompt: The system prompt to include in API-based chat calls.
    
    Returns:
      The generated response(s) as a string or a list of strings.
    """
    if model_config is None:
        return "❌ Error: Invalid model configuration. Please check the model name."
    
    is_batch = isinstance(prompts, list)
    responses = []
    num_input_tokens = 0.0
    num_output_tokens = 0.0

    # Suppress stderr output during execution
    with warnings.catch_warnings(), contextlib.redirect_stderr(None):
        warnings.simplefilter("ignore", category=UserWarning)

        # --- API-based models (OpenAI, Gemini) ---
        if model_config.api_type in ["openai", "gemini"]:
            try:
                messages = [{"role": "system", "content": system_prompt}]
                for prompt in ([prompts] if not is_batch else prompts):
                    user_message = {"role": "user", "content": prompt}
                    full_messages = messages + [user_message]
                    response = model_config.client.chat.completions.create(
                        model=model_config.model_str,
                        messages=full_messages,
                        temperature=temperature,
                        max_tokens=max_new_tokens
                    )
                    response_text = response.choices[0].message.content.strip()
                    responses.append(clean_response(response_text, prompt))
                    if estimate_cost and model_config.cost_per_M_input is not None and model_config.cost_per_M_output is not None:
                        num_input_tokens += response.usage.prompt_tokens
                        num_output_tokens += response.usage.completion_tokens

                if estimate_cost:
                    total_cost = ((num_input_tokens / 1_000_000) * model_config.cost_per_M_input) + \
                                 ((num_output_tokens / 1_000_000) * model_config.cost_per_M_output)
                    print(f"💰 Estimated Cost: ${total_cost:.6f} (Input: {num_input_tokens} tokens, Output: {num_output_tokens} tokens)")
                return responses if is_batch else responses[0]
            except Exception as e:
                return f"{model_config.api_type.capitalize()} API error: {str(e)}"

        # --- Local Hugging Face model generation ---
        tokenizer = model_config.tokenizer
        model = model_config.model

        if model is None or tokenizer is None:
            return "❌ Error: Model or tokenizer is not properly initialized."
        
        # Check if a chat template is available; if not, fallback to normal tokenization.
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
            conversations = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
                            for p in (prompts if is_batch else [prompts])]
            input_ids = tokenizer.apply_chat_template(conversations, return_tensors="pt", padding=True, truncation=True).to(model.device)
        else:
            input_ids = tokenizer(prompts if is_batch else [prompts], return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": terminators,
            "repetition_penalty": 1.2,
            "num_beams": num_beams,
            "do_sample": temperature > 0,
            "temperature": temperature,
        }
        if search_strategy == "top_k":
            gen_kwargs.update({"do_sample": True, "top_k": top_k, "temperature": temperature})
        elif search_strategy == "top_p":
            gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})
        elif search_strategy == "contrastive":
            gen_kwargs.update({"do_sample": True, "penalty_alpha": 0.6, "top_k": 4, "temperature": temperature})

        with torch.no_grad():
            output = model.generate(input_ids, **gen_kwargs)
        responses = tokenizer.batch_decode(output, skip_special_tokens=True)
        responses = [clean_response(resp, prompt) for resp, prompt in zip(responses, prompts if is_batch else [prompts])]

    return responses if is_batch else responses[0]
'''

def llm_generate(model_config, prompts, max_new_tokens=200, temperature=0.7, 
                 search_strategy="top_p", top_k=50, top_p=0.9, num_beams=1,
                 estimate_cost=False, system_prompt="You are an AI assistant that provides brief answers."):
    """
    Generates a response from an LLM using a provided ModelConfig object.
    
    Parameters:
      - model_config: A preconfigured ModelConfig instance.
      - prompts: A single prompt (str) or a list of prompt strings.
      - max_new_tokens, temperature, search_strategy, top_k, top_p, num_beams: Generation parameters.
      - estimate_cost: If True and the cost parameters are set in ModelConfig, prints an estimated cost.
      - system_prompt: The system prompt to include in API-based chat calls.
    
    Returns:
      The cleaned response(s) as a string or a list of strings.
    """
    if model_config is None:
        return "❌ Error: Invalid model configuration. Please check the model name."
    
    is_batch = isinstance(prompts, list)
    responses = []
    num_input_tokens = 0.0
    num_output_tokens = 0.0

    # Suppress stderr output during execution
    with warnings.catch_warnings(), contextlib.redirect_stderr(None):
        warnings.simplefilter("ignore", category=UserWarning)

        # --- API-based models (OpenAI, Gemini) ---
        if model_config.api_type in ["openai", "gemini"]:
            try:
                messages = [{"role": "system", "content": system_prompt}]
                for prompt in ([prompts] if not is_batch else prompts):
                    user_message = {"role": "user", "content": prompt}
                    full_messages = messages + [user_message]
                    response = model_config.client.chat.completions.create(
                        model=model_config.model_str,
                        messages=full_messages,
                        temperature=temperature,
                        max_tokens=max_new_tokens
                    )
                    response_text = response.choices[0].message.content.strip()
                    responses.append(clean_response(response_text, prompt))  # Apply `clean_response`
                    
                    if estimate_cost and model_config.cost_per_M_input is not None and model_config.cost_per_M_output is not None:
                        num_input_tokens += response.usage.prompt_tokens
                        num_output_tokens += response.usage.completion_tokens

                if estimate_cost:
                    total_cost = ((num_input_tokens / 1_000_000) * model_config.cost_per_M_input) + \
                                 ((num_output_tokens / 1_000_000) * model_config.cost_per_M_output)
                    print(f"💰 Estimated Cost: ${total_cost:.6f} (Input: {num_input_tokens} tokens, Output: {num_output_tokens} tokens)")

                return responses if is_batch else responses[0]
            
            except Exception as e:
                return f"{model_config.api_type.capitalize()} API error: {str(e)}"

        # --- Local Hugging Face model generation ---
        tokenizer = model_config.tokenizer
        model = model_config.model

        if model is None or tokenizer is None:
            return "❌ Error: Model or tokenizer is not properly initialized."
        
        # Check if a chat template is available; if not, fallback to normal tokenization.
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
            conversations = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
                            for p in (prompts if is_batch else [prompts])]
            input_ids = tokenizer.apply_chat_template(conversations, return_tensors="pt", padding=True, truncation=True).to(model.device)
        else:
            input_ids = tokenizer(prompts if is_batch else [prompts], return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": terminators,
            "repetition_penalty": 1.2,
            "num_beams": num_beams,
            "do_sample": temperature > 0,
            "temperature": temperature,
        }
        if search_strategy == "top_k":
            gen_kwargs.update({"do_sample": True, "top_k": top_k, "temperature": temperature})
        elif search_strategy == "top_p":
            gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})
        elif search_strategy == "contrastive":
            gen_kwargs.update({"do_sample": True, "penalty_alpha": 0.6, "top_k": 4, "temperature": temperature})

        with torch.no_grad():
            output = model.generate(input_ids, **gen_kwargs)
        responses = tokenizer.batch_decode(output, skip_special_tokens=True)

        # Apply `clean_response` to each response in batch mode
        responses = [clean_response(resp, prompt) for resp, prompt in zip(responses, prompts if is_batch else [prompts])]

    return responses if is_batch else responses[0]



def clear_pipeline(pipe):

    if next(pipe.model.parameters()).is_cuda:
        initial_memory = torch.cuda.memory_allocated() / 1e6
        pipe.model.to("cpu")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated() / 1e6
        print(f"Pipeline cleared.  Freed {initial_memory - final_memory:.2f} MB of CUDA memory")
    else:
        del pipe
        gc.collect()
        print("Pipeline cleared from CPU.")

# helper function to print model info
def print_pipeline_info(pipe):
    model = pipe.model
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.name_or_path}, Size: {model_size:,} parameters")


def display_markdown(response):
    """
    Displays the given text response using Markdown formatting.

    Parameters:
    - response (str): The text response to format as Markdown.
    """
    if not response or not isinstance(response, str):
        display(Markdown("⚠️ *No valid response to display.*"))
        return
    display(Markdown(response))


class JupyterChat:
    def __init__(self, model_str, system_prompt="You are a helpful assistant."):
        """
        Initializes a Jupyter-based chat interface using llm_configure and llm_generate.

        Parameters:
        - model_str (str): The model to load (can be a local or API-based model).
        - system_prompt (str): The system message that sets the behavior of the assistant.
        """
        self.model_config = llm_configure(model_str)
        if not self.model_config:
            raise ValueError(f"Could not load model: {model_str}")
        
        self.system_prompt = system_prompt
        self.chat_history = []
        
        # Initialize UI elements
        self.text_input = widgets.Text(
            placeholder="Type your message here...",
            description="User:",
            layout=widgets.Layout(width="100%")
        )
        self.send_button = widgets.Button(description="Send", button_style='primary')
        self.clear_button = widgets.Button(description="Clear Chat", button_style='warning')
        self.copy_button = widgets.Button(description="Copy", button_style='success')
        self.output_area = widgets.Output()

        # Event handlers
        self.send_button.on_click(self.handle_input)
        self.text_input.on_submit(self.handle_input)
        self.clear_button.on_click(self.clear_chat)
        self.copy_button.on_click(self.copy_chat)

        # Layout the buttons
        button_box = widgets.HBox([self.send_button, self.clear_button, self.copy_button])

        # Display chat UI
        display(self.text_input, button_box, self.output_area)

    def handle_input(self, event):
        """Handles user input and updates the chat history."""
        user_input = self.text_input.value.strip()
        if not user_input:
            return
        
        self.text_input.value = ""  # Clear input field
        self.chat_history.append({"role": "user", "content": user_input})

        response = self.generate_response(user_input)
        self.chat_history.append({"role": "assistant", "content": response})
        
        self.update_display()  # Refresh output display

    def generate_response(self, user_input):
        """Generates a response using llm_generate."""
        try:
            response = llm_generate(self.model_config, prompts=user_input, system_prompt=self.system_prompt)
            return response
        except Exception as e:
            return self.format_api_error(e)

    def update_display(self):
        """Updates the chat output to prevent duplicate messages and ensure correct Markdown rendering."""
        with self.output_area:
            clear_output()
            for entry in self.chat_history:
                role = "**User:**" if entry["role"] == "user" else "**Assistant:**"
                message = f"{role}\n\n{entry['content']}"
                display_markdown(message)

    def clear_chat(self, event):
        """Clears the chat history and output display."""
        self.chat_history = []
        with self.output_area:
            clear_output()
        print("Chat history cleared.")

    def copy_chat(self, event):
        """Copies the raw chat history (unformatted) to the clipboard."""
        chat_text = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in self.chat_history])
        pyperclip.copy(chat_text)
        print("✅ Chat copied to clipboard!")

    def format_api_error(self, error):
        """Formats API error messages for better readability."""
        error_str = str(error)
        if "429" in error_str:
            return "⚠️ *API limit reached. Please check your quota.*"
        elif "401" in error_str:
            return "⚠️ *Invalid API key. Please verify your credentials.*"
        elif "RESOURCE_EXHAUSTED" in error_str:
            return "⚠️ *Quota exceeded. Try again later.*"
        return f"⚠️ *API Error:* {error_str}"

# Example usage:
# chat = JupyterChat("gemini-2.0-flash")
