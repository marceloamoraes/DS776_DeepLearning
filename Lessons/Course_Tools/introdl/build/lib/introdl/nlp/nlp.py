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
import time
from tqdm.autonotebook import tqdm
import sys

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
SHORT_NAME_MAP = {
    "llama-3p1-8B": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "mistral-7B": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "llama-3p2-3B": "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    "gemini-flash-lite": "gemini-2.0-flash-lite-preview-02-05",
    "gemini-flash": "gemini-2.0-flash",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "o1-mini": "o1-mini",
    "o3-mini": "o3-mini",
}

OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini"]
GEMINI_MODELS = ["gemini-2.0-flash-lite-preview-02-05", "gemini-2.0-flash"]

def llm_list_models(verbose=True):
    """
    Lists all available models grouped by category:
    - Short Name Models (Hugging Face & Gemini)
    - OpenAI Models
    - Gemini Models
    """
    if verbose:
        print("Available models:")
        for short_name in SHORT_NAME_MAP.keys():
            if short_name in OPENAI_MODELS:
                print(f" {short_name} => needs OPENAI_API_KEY")
            elif short_name.startswith('gemini'):
                print(f" {short_name} => needs GEMINI_API_KEY")
            else:
                print(f" {short_name} => HuggingFace: {SHORT_NAME_MAP[short_name]}")
    models = zip(range(len(SHORT_NAME_MAP)), SHORT_NAME_MAP.keys())
    return models

def release_model():
    """Unloads the currently loaded Hugging Face model from memory and GPU."""
    global LOADED_MODELS, LOADED_TOKENIZERS
    if LOADED_MODELS:
        model_str = next(iter(LOADED_MODELS))
        print(f"🛑 Unloading model: {model_str} from GPU...")
        model = LOADED_MODELS[model_str]
        try:
            if hasattr(model, "hf_device_map"):
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
    
    When using API-based models (e.g. OpenAI, Gemini, Together, or Groq), the model and tokenizer
    may be None.
    """
    def __init__(self, model_str, model=None, tokenizer=None, api_type=None,
                 cost_per_M_input=None, cost_per_M_output=None):
        self.model_str = model_str
        self.model = model
        self.tokenizer = tokenizer
        self.api_type = api_type  # Expected values: "openai", "gemini", "together", "groq" or None
        self.client = None
        self.cost_per_M_input = cost_per_M_input
        self.cost_per_M_output = cost_per_M_output

        # Initialize API client if applicable.
        if api_type in ["openai", "gemini", "together", "groq"]:
            if api_type == "together":
                api_key = os.getenv("TOGETHER_API_KEY")
                base_url = "https://api.together.xyz/v1"
            elif api_type == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                base_url = "https://api.groq.com/openai/v1"
            else:
                api_key = os.getenv("GEMINI_API_KEY" if api_type == "gemini" else "OPENAI_API_KEY")
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai/" if api_type == "gemini" else None
            if not api_key:
                raise ValueError(f"Missing {api_type.upper()} API key. Set the appropriate environment variable.")
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def unload(self):
        release_model()

def llm_configure(model_str, cost_per_M_input=None, cost_per_M_output=None,
                  llm_provider=None):
    """
    Configures the model based on the provided model_str.
    In addition to cost parameters, this function allows overriding the provider via the `llm_provider` argument.
    
    Parameters:
      - cost_per_M_input: Cost per million input tokens (for API-based models).
      - cost_per_M_output: Cost per million output tokens (for API-based models).
      - llm_provider (str, optional): Can be set to "together" or "groq" to force the use of the corresponding API.
      - model_str: A model name string (can be a short alias or a full model name).
    
    Returns:
      A ModelConfig object.
    """
    model_str = model_str.strip()
    
    # Validate llm_provider if provided.
    if llm_provider is not None and llm_provider not in ["together", "groq"]:
        raise ValueError("llm_provider must be either None, 'together', or 'groq'")
    
    # Expand short names.
    if model_str in SHORT_NAME_MAP:
        model_str = SHORT_NAME_MAP[model_str]
    
    # If llm_provider is provided, override API selection.
    if llm_provider is not None:
        return ModelConfig(model_str, api_type=llm_provider,
                           cost_per_M_input=cost_per_M_input, cost_per_M_output=cost_per_M_output)
    
    # For API-based models.
    if model_str in OPENAI_MODELS:
        return ModelConfig(model_str, api_type="openai",
                           cost_per_M_input=cost_per_M_input, cost_per_M_output=cost_per_M_output)
    if model_str in GEMINI_MODELS:
        return ModelConfig(model_str, api_type="gemini",
                           cost_per_M_input=cost_per_M_input, cost_per_M_output=cost_per_M_output)
    
    # For Hugging Face models: use the global cache.
    global LOADED_MODELS, LOADED_TOKENIZERS
    if model_str in LOADED_MODELS:
        return ModelConfig(model_str, LOADED_MODELS[model_str], LOADED_TOKENIZERS[model_str])
    elif LOADED_MODELS:
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
        return ModelConfig(model_str, model, tokenizer,
                           cost_per_M_input=cost_per_M_input, cost_per_M_output=cost_per_M_output)
    except Exception as e:
        print(f"❌ Error loading model {model_str}: {str(e)}")
        return None

def clean_response(response, prompt=None):
    """
    Cleans the response by removing the input prompt, an 'assistant' label,
    and any extraneous blank lines.

    Additionally, if a prompt is provided, it searches for the last 10 characters
    of the prompt concatenated with "assistant", ignoring whitespace differences,
    and if found, removes everything in the response up to and including that string.
    """
    if prompt:
        # --- Original prompt removal ---
        prompt_marker_pattern = re.escape(prompt) + r"\s*\.\s*assistant"
        match = re.search(prompt_marker_pattern, response)
        if match:
            response = response[match.end():].strip()
        else:
            match = re.search(re.escape(prompt), response)
            if match:
                response = response[match.end():].strip()

        # --- Additional marker removal ---
        extra_marker = prompt[-10:] + "assistant"
        # Build a regex pattern that ignores whitespace differences.
        pattern = ""
        for char in extra_marker:
            if char.isspace():
                pattern += r"\s+"
            else:
                pattern += re.escape(char) + r"\s*"
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            response = response[match.end():].strip()

    # Remove any lines that consist solely of the word "assistant" (ignoring case).
    response = re.sub(r"^\s*assistant\s*", "", response, flags=re.IGNORECASE).strip()
    response = "\n".join([line for line in response.split("\n") if line.strip()])
    return response

'''
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
    return response'
'''


def llm_generate(model_config, prompts, 
                 batch_size=1, assistant_prompt=None, estimate_cost=False, max_new_tokens=200, 
                 remove_input_prompt=True, search_strategy="top_p", top_k=50, top_p=0.9, num_beams=1, temperature=0.7,
                 system_prompt="You are an AI assistant that provides brief answers. Do not include the input prompt in your answer.",
                 rate_limit=None,
                 disable_tqdm=False):
    """
    Generates responses from a language model.
    
    This function handles search strategy and parameters on the fly and uses progress bars
    for batch processing (both for API-based and local generation) if disable_tqdm is False.
    
    Parameters (in alphabetical order):
      - assistant_prompt (str, optional): An optional previous assistant message to include in the conversation.
      - batch_size (int): Number of prompts processed per batch. (default: 1)
      - estimate_cost (bool): If True, prints estimated token cost (if cost parameters are set). (default: False)
      - max_new_tokens (int): Maximum number of new tokens to generate. (default: 200)
      - num_beams (int): Number of beams for beam search. (default: 1)
      - remove_input_prompt (bool): If True, removes the input prompt from the generated response. (default: True)
      - search_strategy (str): Decoding strategy; supported options: "top_k", "top_p", "contrastive", "beam_search", "deterministic". (default: "top_p")
      - system_prompt (str): System-level instruction prompt for API-based models. (default updated for cleaner outputs)
      - temperature (float): Sampling temperature. (default: 0.7)
      - top_k (int): Number of top tokens to consider for "top_k" strategy. (default: 50)
      - top_p (float): Cumulative probability threshold for nucleus sampling ("top_p"). (default: 0.9)
      - rate_limit (float, optional): Maximum number of API requests per minute. If provided, the function will delay API calls accordingly.
      - disable_tqdm (bool, optional): If True, disables tqdm progress bars. (default: False)
    
    Additional required parameters:
      - model_config (ModelConfig): Preconfigured instance.
      - prompts (str or list of str): Single prompt or batch of prompts.
    
    Returns:
      A generated response (or list of responses) as a string.
    """
    # Use search_strategy directly.
    strategy = search_strategy

    if model_config is None:
        return "❌ Error: Invalid model configuration. Please check the model name."
    
    is_batch = isinstance(prompts, list)
    prompt_list = prompts if is_batch else [prompts]
    responses = []
    num_input_tokens = 0.0
    num_output_tokens = 0.0

    # --- API-based models ---
    if model_config.api_type in ["openai", "gemini", "together", "groq"]:
        try:
            messages = [{"role": "system", "content": system_prompt}]
            if assistant_prompt is not None:
                messages.append({"role": "assistant", "content": assistant_prompt})
            
            with tqdm(total=len(prompt_list), desc="API Generation", disable=disable_tqdm,
                      leave=True, dynamic_ncols=True, file=sys.stdout) as pbar:
                for prompt in prompt_list:
                    user_message = {"role": "user", "content": prompt}
                    full_messages = messages + [user_message]

                    openai_params = {
                        "model": model_config.model_str,
                        "messages": full_messages,
                        "max_tokens": max_new_tokens
                    }
                    if strategy == "deterministic":
                        openai_params["temperature"] = 0.0
                        openai_params["top_p"] = 1.0
                    else:
                        openai_params["temperature"] = temperature
                        openai_params["top_p"] = top_p

                    response = model_config.client.chat.completions.create(**openai_params)
                    response_text = response.choices[0].message.content.strip()
                    responses.append(clean_response(response_text, prompt) if remove_input_prompt else response_text)
                    
                    if rate_limit is not None:
                        time.sleep(60.0 / rate_limit)
                    
                    if estimate_cost and model_config.cost_per_M_input is not None and model_config.cost_per_M_output is not None:
                        num_input_tokens += response.usage.prompt_tokens
                        num_output_tokens += response.usage.completion_tokens

                    pbar.update(1)
            
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
    
    def process_batch(batch_prompts):
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
            conversations = []
            for p in batch_prompts:
                conv = [{"role": "system", "content": system_prompt}]
                if assistant_prompt is not None:
                    conv.append({"role": "assistant", "content": assistant_prompt})
                conv.append({"role": "user", "content": p})
                conversations.append(conv)
            input_ids = tokenizer.apply_chat_template(conversations, return_tensors="pt",
                                                      padding=True, truncation=True).to(model.device)
        else:
            new_prompts = []
            for p in batch_prompts:
                prompt_parts = [system_prompt]
                if assistant_prompt is not None:
                    prompt_parts.append("Assistant: " + assistant_prompt)
                prompt_parts.append("User: " + p)
                new_prompts.append("\n".join(prompt_parts))
            input_ids = tokenizer(new_prompts, return_tensors="pt",
                                  padding=True, truncation=True).to(model.device)

        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": terminators,
            "repetition_penalty": 1.2,
        }
        if strategy == "top_k":
            gen_kwargs.update({"do_sample": True, "top_k": top_k, "temperature": temperature})
        elif strategy == "top_p":
            gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})
        elif strategy == "contrastive":
            gen_kwargs.update({"do_sample": True, "penalty_alpha": 0.6, "top_k": 4, "temperature": temperature})
        elif strategy == "beam_search":
            gen_kwargs.update({"do_sample": False, "num_beams": num_beams, "temperature": temperature})
        elif strategy == "deterministic":
            gen_kwargs.update({"do_sample": False, "temperature": 0.0, "num_beams": 1})
        else:
            gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})

        with torch.no_grad():
            output = model.generate(input_ids, **gen_kwargs)
        batch_responses = tokenizer.batch_decode(output, skip_special_tokens=True)
        if remove_input_prompt:
            return [clean_response(resp, prompt) for resp, prompt in zip(batch_responses, batch_prompts)]
        else:
            return batch_responses

    with tqdm(total=len(prompt_list), desc="Local Generation", disable=disable_tqdm,
              leave=True, dynamic_ncols=True, file=sys.stdout) as pbar:
        all_responses = []
        for i in range(0, len(prompt_list), batch_size):
            batch_prompts = prompt_list[i:i + batch_size]
            all_responses.extend(process_batch(batch_prompts))
            pbar.update(len(batch_prompts))
        responses = all_responses

    return responses if is_batch else responses[0]


'''
def llm_generate(model_config, prompts, 
                 batch_size=1, assistant_prompt=None, estimate_cost=False, max_new_tokens=200, 
                 remove_input_prompt=True, search_strategy="top_p", top_k=50, top_p=0.9, num_beams=1, temperature=0.7,
                 system_prompt="You are an AI assistant that provides brief answers. Do not include the input prompt in your answer.",
                 rate_limit=None):
    """
    Generates responses from a language model.
    
    This function handles search strategy and parameters on the fly.
    
    Parameters (in alphabetical order):
      - assistant_prompt (str, optional): An optional previous assistant message to include in the conversation.
      - batch_size (int): Number of prompts processed per batch. (default: 1)
      - estimate_cost (bool): If True, prints estimated token cost (if cost parameters are set). (default: False)
      - max_new_tokens (int): Maximum number of new tokens to generate. (default: 200)
      - num_beams (int): Number of beams for beam search. (default: 1)
      - remove_input_prompt (bool): If True, removes the input prompt from the generated response. (default: True)
      - search_strategy (str): Decoding strategy; supported options: "top_k", "top_p", "contrastive", "beam_search", "deterministic". (default: "top_p")
      - system_prompt (str): System-level instruction prompt for API-based models. (default updated for cleaner outputs)
      - temperature (float): Sampling temperature. (default: 0.7)
      - top_k (int): Number of top tokens to consider for "top_k" strategy. (default: 50)
      - top_p (float): Cumulative probability threshold for nucleus sampling ("top_p"). (default: 0.9)
      - rate_limit (float, optional): Maximum number of API requests per minute. If provided, the function will delay API calls accordingly.
    
    Additional required parameters:
      - model_config (ModelConfig): Preconfigured instance.
      - prompts (str or list of str): Single prompt or batch of prompts.
    
    Returns:
      A generated response (or list of responses) as a string.
    """
    # Use search_strategy directly.
    strategy = search_strategy

    if model_config is None:
        return "❌ Error: Invalid model configuration. Please check the model name."
    
    is_batch = isinstance(prompts, list)
    prompt_list = prompts if is_batch else [prompts]
    responses = []
    num_input_tokens = 0.0
    num_output_tokens = 0.0

    with warnings.catch_warnings(), contextlib.redirect_stderr(None):
        warnings.simplefilter("ignore", category=UserWarning)

        # --- API-based models (OpenAI, Gemini, Together, Groq) ---
        if model_config.api_type in ["openai", "gemini", "together", "groq"]:
            try:
                # Build initial message list with system prompt and optional assistant_prompt.
                messages = [{"role": "system", "content": system_prompt}]
                if assistant_prompt is not None:
                    messages.append({"role": "assistant", "content": assistant_prompt})
                for prompt in prompt_list:
                    user_message = {"role": "user", "content": prompt}
                    full_messages = messages + [user_message]

                    openai_params = {
                        "model": model_config.model_str,
                        "messages": full_messages,
                        "max_tokens": max_new_tokens
                    }
                    if strategy == "deterministic":
                        openai_params["temperature"] = 0.0
                        openai_params["top_p"] = 1.0
                    else:
                        openai_params["temperature"] = temperature
                        openai_params["top_p"] = top_p

                    response = model_config.client.chat.completions.create(**openai_params)
                    response_text = response.choices[0].message.content.strip()
                    responses.append(clean_response(response_text, prompt) if remove_input_prompt else response_text)
                    
                    # If rate_limit is specified, sleep to ensure we do not exceed the limit.
                    if rate_limit is not None:
                        delay = 60.0 / rate_limit
                        time.sleep(delay)
                    
                    if estimate_cost:
                        num_input_tokens += response.usage.prompt_tokens
                        num_output_tokens += response.usage.completion_tokens

                if estimate_cost:
                    if model_config.cost_per_M_input is not None and model_config.cost_per_M_output is not None:
                        total_cost = ((num_input_tokens / 1_000_000) * model_config.cost_per_M_input) + \
                                    ((num_output_tokens / 1_000_000) * model_config.cost_per_M_output)
                        print(f"💰 Estimated Cost: ${total_cost:.6f} (Input: {num_input_tokens} tokens, Output: {num_output_tokens} tokens)")
                    else:
                        print("⚠️ Cost parameters not set. Cannot estimate cost.")
                        print(f"💰 Token Usage: Input: {num_input_tokens} tokens, Output: {num_output_tokens} tokens")

                return responses if is_batch else responses[0]
            
            except Exception as e:
                return f"{model_config.api_type.capitalize()} API error: {str(e)}"

        # --- Local Hugging Face model generation ---
        tokenizer = model_config.tokenizer
        model = model_config.model

        if model is None or tokenizer is None:
            return "❌ Error: Model or tokenizer is not properly initialized."
        
        def process_batch(batch_prompts):
            # If a chat template is available, build conversation including system prompt, optional assistant_prompt, and user.
            if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
                conversations = []
                for p in batch_prompts:
                    conv = [{"role": "system", "content": system_prompt}]
                    if assistant_prompt is not None:
                        conv.append({"role": "assistant", "content": assistant_prompt})
                    conv.append({"role": "user", "content": p})
                    conversations.append(conv)
                input_ids = tokenizer.apply_chat_template(conversations, return_tensors="pt",
                                                          padding=True, truncation=True).to(model.device)
            else:
                # No chat template: build a prompt string including system prompt and optional assistant_prompt.
                new_prompts = []
                for p in batch_prompts:
                    prompt_parts = [system_prompt]
                    if assistant_prompt is not None:
                        prompt_parts.append("Assistant: " + assistant_prompt)
                    prompt_parts.append("User: " + p)
                    new_prompts.append("\n".join(prompt_parts))
                input_ids = tokenizer(new_prompts, return_tensors="pt",
                                      padding=True, truncation=True).to(model.device)

            terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "eos_token_id": terminators,
                "repetition_penalty": 1.2,
            }
            if strategy == "top_k":
                gen_kwargs.update({"do_sample": True, "top_k": top_k, "temperature": temperature})
            elif strategy == "top_p":
                gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})
            elif strategy == "contrastive":
                gen_kwargs.update({"do_sample": True, "penalty_alpha": 0.6, "top_k": 4, "temperature": temperature})
            elif strategy == "beam_search":
                gen_kwargs.update({"do_sample": False, "num_beams": num_beams, "temperature": temperature})
            elif strategy == "deterministic":
                gen_kwargs.update({"do_sample": False, "temperature": 0.0, "num_beams": 1})
            else:
                gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})

            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)
            
            batch_responses = tokenizer.batch_decode(output, skip_special_tokens=True)
            if remove_input_prompt:
                return [clean_response(resp, prompt) for resp, prompt in zip(batch_responses, batch_prompts)]
            else:
                return batch_responses

        all_responses = []
        for i in range(0, len(prompt_list), batch_size):
            batch_prompts = prompt_list[i:i + batch_size]
            all_responses.extend(process_batch(batch_prompts))
        responses = all_responses

    return responses if is_batch else responses[0]
'''


'''
def llm_generate(model_config, prompts, 
                 batch_size=1, assistant_prompt=None, estimate_cost=False, max_new_tokens=200, 
                 remove_input_prompt=True, search_strategy="top_p", top_k=50, top_p=0.9, num_beams=1, temperature=0.7,
                 system_prompt="You are an AI assistant that provides brief answers."):
    """
    Generates responses from a language model.
    
    This function handles search strategy and parameters on the fly.
    
    Parameters (in alphabetical order):
      - assistant_prompt (str, optional): An optional previous assistant message to include in the conversation.
      - batch_size (int): Number of prompts processed per batch. (default: 1)
      - estimate_cost (bool): If True, prints estimated token cost (if cost parameters are set). (default: False)
      - max_new_tokens (int): Maximum number of new tokens to generate. (default: 200)
      - num_beams (int): Number of beams for beam search. (default: 1)
      - remove_input_prompt (bool): If True, removes the input prompt from the generated response. (default: True)
      - search_strategy (str): Decoding strategy; supported options: "top_k", "top_p", "contrastive", "beam_search", "deterministic". (default: "top_p")
      - system_prompt (str): System-level instruction prompt for API-based models. (default: "You are an AI assistant that provides brief answers.")
      - temperature (float): Sampling temperature. (default: 0.7)
      - top_k (int): Number of top tokens to consider for "top_k" strategy. (default: 50)
      - top_p (float): Cumulative probability threshold for nucleus sampling ("top_p"). (default: 0.9)
    
    Additional required parameters:
      - model_config (ModelConfig): Preconfigured instance.
      - prompts (str or list of str): Single prompt or batch of prompts.
    
    Returns:
      A generated response (or list of responses) as a string.
    """
    # Use search_strategy directly.
    strategy = search_strategy

    if model_config is None:
        return "❌ Error: Invalid model configuration. Please check the model name."
    
    is_batch = isinstance(prompts, list)
    prompt_list = prompts if is_batch else [prompts]
    responses = []
    num_input_tokens = 0.0
    num_output_tokens = 0.0

    with warnings.catch_warnings(), contextlib.redirect_stderr(None):
        warnings.simplefilter("ignore", category=UserWarning)

        # --- API-based models (OpenAI, Gemini, Together, Groq) ---
        if model_config.api_type in ["openai", "gemini", "together", "groq"]:
            try:
                # Build initial message list with system prompt and optional assistant_prompt.
                messages = [{"role": "system", "content": system_prompt}]
                if assistant_prompt is not None:
                    messages.append({"role": "assistant", "content": assistant_prompt})
                for prompt in prompt_list:
                    user_message = {"role": "user", "content": prompt}
                    full_messages = messages + [user_message]

                    openai_params = {
                        "model": model_config.model_str,
                        "messages": full_messages,
                        "max_tokens": max_new_tokens
                    }
                    if strategy == "deterministic":
                        openai_params["temperature"] = 0.0
                        openai_params["top_p"] = 1.0
                    else:
                        openai_params["temperature"] = temperature
                        openai_params["top_p"] = top_p

                    response = model_config.client.chat.completions.create(**openai_params)
                    response_text = response.choices[0].message.content.strip()
                    responses.append(clean_response(response_text, prompt) if remove_input_prompt else response_text)
                    
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
        
        def process_batch(batch_prompts):
            # If a chat template is available, build conversation including system prompt, optional assistant_prompt, and user.
            if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
                conversations = []
                for p in batch_prompts:
                    conv = [{"role": "system", "content": system_prompt}]
                    if assistant_prompt is not None:
                        conv.append({"role": "assistant", "content": assistant_prompt})
                    conv.append({"role": "user", "content": p})
                    conversations.append(conv)
                input_ids = tokenizer.apply_chat_template(conversations, return_tensors="pt",
                                                          padding=True, truncation=True).to(model.device)
            else:
                # No chat template: build a prompt string including system prompt and optional assistant_prompt.
                new_prompts = []
                for p in batch_prompts:
                    prompt_parts = [system_prompt]
                    if assistant_prompt is not None:
                        prompt_parts.append("Assistant: " + assistant_prompt)
                    prompt_parts.append("User: " + p)
                    new_prompts.append("\n".join(prompt_parts))
                input_ids = tokenizer(new_prompts, return_tensors="pt",
                                      padding=True, truncation=True).to(model.device)

            terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "eos_token_id": terminators,
                "repetition_penalty": 1.2,
            }
            if strategy == "top_k":
                gen_kwargs.update({"do_sample": True, "top_k": top_k, "temperature": temperature})
            elif strategy == "top_p":
                gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})
            elif strategy == "contrastive":
                gen_kwargs.update({"do_sample": True, "penalty_alpha": 0.6, "top_k": 4, "temperature": temperature})
            elif strategy == "beam_search":
                gen_kwargs.update({"do_sample": False, "num_beams": num_beams, "temperature": temperature})
            elif strategy == "deterministic":
                gen_kwargs.update({"do_sample": False, "temperature": 0.0, "num_beams": 1})
            else:
                gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})

            with torch.no_grad():
                output = model.generate(input_ids, **gen_kwargs)
            
            batch_responses = tokenizer.batch_decode(output, skip_special_tokens=True)
            if remove_input_prompt:
                return [clean_response(resp, prompt) for resp, prompt in zip(batch_responses, batch_prompts)]
            else:
                return batch_responses

        all_responses = []
        for i in range(0, len(prompt_list), batch_size):
            batch_prompts = prompt_list[i:i + batch_size]
            all_responses.extend(process_batch(batch_prompts))
        responses = all_responses

    return responses if is_batch else responses[0]'
'''

def clear_pipeline(pipe, verbosity=0):
    """Clears a Hugging Face pipeline and frees CUDA memory."""
    if hasattr(pipe, "model") and next(pipe.model.parameters()).is_cuda:
        initial_allocated = torch.cuda.memory_allocated() / 1e6
        initial_reserved = torch.cuda.memory_reserved() / 1e6

        if verbosity > 0:
            print(f"🔍 Before unloading: {initial_allocated:.2f} MB allocated, {initial_reserved:.2f} MB reserved.")

        try:
            pipe.model.to("cpu")
            for param in pipe.model.parameters():
                param.data = param.data.cpu()
        except Exception as e:
            if verbosity > 0:
                print(f"⚠️ Error moving model to CPU: {e}")

        del pipe.model
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        final_allocated = torch.cuda.memory_allocated() / 1e6
        final_reserved = torch.cuda.memory_reserved() / 1e6

        if verbosity > 0:
            print(f"✅ Pipeline cleared. Freed {initial_allocated - final_allocated:.2f} MB allocated, "
                  f"{initial_reserved - final_reserved:.2f} MB reserved.")
    else:
        if verbosity > 0:
            print("ℹ️ Pipeline already on CPU. Performing standard cleanup.")
        del pipe
        gc.collect()

    if verbosity > 0:
        print("🗑️ Cleanup complete.")
    elif verbosity == 0:
        print("✅ Pipeline cleared.")

def print_pipeline_info(pipe):
    model = pipe.model
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.name_or_path}, Size: {model_size:,} parameters")

def display_markdown(response):
    if not response or not isinstance(response, str):
        display(Markdown("⚠️ *No valid response to display.*"))
        return
    display(Markdown(response))

class JupyterChat:
    def __init__(self, model_str, system_prompt="You are a helpful assistant."):
        self.model_config = llm_configure(model_str)
        if not self.model_config:
            raise ValueError(f"Could not load model: {model_str}")
        
        self.system_prompt = system_prompt
        self.chat_history = []
        
        self.text_input = widgets.Text(
            placeholder="Type your message here...",
            description="User:",
            layout=widgets.Layout(width="100%")
        )
        self.send_button = widgets.Button(description="Send", button_style='primary')
        self.clear_button = widgets.Button(description="Clear Chat", button_style='warning')
        self.copy_button = widgets.Button(description="Copy", button_style='success')
        self.output_area = widgets.Output()

        self.send_button.on_click(self.handle_input)
        self.text_input.on_submit(self.handle_input)
        self.clear_button.on_click(self.clear_chat)
        self.copy_button.on_click(self.copy_chat)

        button_box = widgets.HBox([self.send_button, self.clear_button, self.copy_button])
        display(self.text_input, button_box, self.output_area)

    def handle_input(self, event):
        user_input = self.text_input.value.strip()
        if not user_input:
            return
        self.text_input.value = ""
        self.chat_history.append({"role": "user", "content": user_input})
        response = self.generate_response(user_input)
        self.chat_history.append({"role": "assistant", "content": response})
        self.update_display()

    def generate_response(self, user_input):
        try:
            response = llm_generate(self.model_config, prompts=user_input, system_prompt=self.system_prompt)
            return response
        except Exception as e:
            return self.format_api_error(e)

    def update_display(self):
        with self.output_area:
            clear_output()
            for entry in self.chat_history:
                role = "**User:**" if entry["role"] == "user" else "**Assistant:**"
                message = f"{role}\n\n{entry['content']}"
                display_markdown(message)

    def clear_chat(self, event):
        self.chat_history = []
        with self.output_area:
            clear_output()
        print("Chat history cleared.")

    def copy_chat(self, event):
        chat_text = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in self.chat_history])
        pyperclip.copy(chat_text)
        print("✅ Chat copied to clipboard!")

    def format_api_error(self, error):
        error_str = str(error)
        if "429" in error_str:
            return "⚠️ *API limit reached. Please check your quota.*"
        elif "401" in error_str:
            return "⚠️ *Invalid API key. Please verify your credentials.*"
        elif "RESOURCE_EXHAUSTED" in error_str:
            return "⚠️ *Quota exceeded. Try again later.*"
        return f"⚠️ *API Error:* {error_str}"
