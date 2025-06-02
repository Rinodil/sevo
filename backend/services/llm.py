"""
LLM API interface for making calls to various language models.

This module provides a unified interface for making API calls to different LLM providers
(OpenAI, Anthropic, Groq, etc.) using LiteLLM. It includes support for:
- Streaming responses
- Tool calls and function calling
- Retry logic with exponential backoff
- Model-specific configurations
- Comprehensive error handling and logging
"""

from typing import Union, Dict, Any, Optional, AsyncGenerator, List
import os
import json
import asyncio
from datetime import datetime # Added
from openai import OpenAIError
import litellm
from utils.logger import logger
from utils.config import config
from services import redis # Added

# litellm.set_verbose=True
litellm.modify_params=True

# Constants
MAX_RETRIES = 2
RATE_LIMIT_DELAY = 30
RETRY_DELAY = 0.1

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class LLMRetryError(LLMError):
    """Exception raised when retries are exhausted."""
    pass

class LLMTokenRateLimitError(LLMError):
    """Exception raised when a token rate limit is exceeded."""
    def __init__(self, message: str, model_name: Optional[str] = None, limit: int = 0, usage: int = 0, retry_after: Optional[int] = None):
        super().__init__(message)
        self.model_name = model_name
        self.limit = limit
        self.usage = usage
        self.retry_after = retry_after

# --- Token Rate Limiting Helper Functions ---

async def get_token_rate_limit(model_name: str) -> Optional[int]:
    """
    Retrieves the token-per-minute (TPM) limit for a given model.

    Args:
        model_name: The name of the model.

    Returns:
        The TPM limit if configured, otherwise None.
    """
    if not config.USER_SPECIFIC_TOKEN_RATE_LIMITS_ENABLED:
        return None

    limits = config.TOKEN_RATE_LIMITS
    
    # Try exact match
    if model_name in limits:
        return limits[model_name].get("tpm")

    # Try cleaned model name (e.g., "openrouter/openai/gpt-4o-mini" -> "gpt-4o-mini")
    cleaned_model_name = model_name.split('/')[-1]
    if cleaned_model_name in limits:
        return limits[cleaned_model_name].get("tpm")
    
    # Fallback to default
    default_limit = limits.get("default", {}).get("tpm")
    if default_limit:
        logger.debug(f"Using default TPM limit for model {model_name}: {default_limit}")
        return default_limit
        
    logger.warning(f"No TPM limit found for model {model_name} or default.")
    return None

async def check_token_rate_limit(account_id: str, model_name: str, estimated_input_tokens: int):
    """
    Checks if the estimated token usage for an account and model exceeds the rate limit.

    Args:
        account_id: The ID of the account.
        model_name: The name of the model.
        estimated_input_tokens: The estimated number of input tokens for the request.

    Raises:
        LLMTokenRateLimitError: If the estimated usage exceeds the limit.
    """
    try:
        tpm_limit = await get_token_rate_limit(model_name)
        if tpm_limit is None:
            logger.debug(f"Token rate limiting not applicable or configured for model {model_name}.")
            return

        redis_client = await redis.get_client()
        now = datetime.now()
        current_minute_timestamp = now.strftime('%Y%m%d%H%M')
        rate_limit_key = f"token_rl:{account_id}:{model_name}:{current_minute_timestamp}"

        current_usage_str = await redis_client.get(rate_limit_key)
        current_usage = int(current_usage_str) if current_usage_str else 0

        if current_usage + estimated_input_tokens > tpm_limit:
            remaining_window_seconds = config.TOKEN_RATE_LIMIT_WINDOW_SECONDS - now.second
            error_msg = (
                f"Token rate limit exceeded for account {account_id} on model {model_name}. "
                f"Limit: {tpm_limit} TPM, Current Usage: {current_usage}, Estimated Input: {estimated_input_tokens}. "
                f"Try again in {remaining_window_seconds} seconds."
            )
            logger.warning(error_msg)
            raise LLMTokenRateLimitError(
                message=error_msg,
                model_name=model_name,
                limit=tpm_limit,
                usage=current_usage,
                retry_after=remaining_window_seconds
            )
        
        logger.debug(
            f"Token rate limit check passed for account {account_id}, model {model_name}. "
            f"Usage: {current_usage + estimated_input_tokens}/{tpm_limit} TPM."
        )

    except redis.RedisError as e:
        logger.error(f"Redis error during token rate limit check for account {account_id}, model {model_name}: {e}. Failing open.")
        # Fail open: If Redis is down, allow the request
    except LLMTokenRateLimitError:
        raise # Re-raise the specific error to be caught by the caller
    except Exception as e:
        logger.error(f"Unexpected error during token rate limit check for account {account_id}, model {model_name}: {e}", exc_info=True)
        # Fail open for unexpected errors as well, or consider stricter policy

async def update_token_usage(account_id: str, model_name: str, prompt_tokens: int, completion_tokens: int):
    """
    Updates the token usage for an account and model in Redis.

    Args:
        account_id: The ID of the account.
        model_name: The name of the model.
        prompt_tokens: The number of tokens in the prompt.
        completion_tokens: The number of tokens in the completion.
    """
    try:
        if not config.USER_SPECIFIC_TOKEN_RATE_LIMITS_ENABLED or not account_id:
            return

        tpm_limit = await get_token_rate_limit(model_name)
        if tpm_limit is None:
            # This means rate limiting is not configured for this model or globally disabled
            return

        total_tokens_used = prompt_tokens + completion_tokens
        if total_tokens_used == 0:
            return

        redis_client = await redis.get_client()
        now = datetime.now()
        current_minute_timestamp = now.strftime('%Y%m%d%H%M')
        rate_limit_key = f"token_rl:{account_id}:{model_name}:{current_minute_timestamp}"
        
        # Increment usage and check if the key was new
        new_value = await redis_client.incrby(rate_limit_key, total_tokens_used)
        
        # If the key was new (i.e., its value is exactly total_tokens_used after incrby), set an expiry
        if new_value == total_tokens_used:
            # Add a small buffer to the expiry to handle clock skew and ensure key exists for the whole minute
            await redis_client.expire(rate_limit_key, config.TOKEN_RATE_LIMIT_WINDOW_SECONDS + 30)

        logger.debug(
            f"Updated token usage for account {account_id}, model {model_name}. "
            f"Added: {total_tokens_used}, New total for current minute: {new_value}."
        )

    except redis.RedisError as e:
        logger.error(f"Redis error during token usage update for account {account_id}, model {model_name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during token usage update for account {account_id}, model {model_name}: {e}", exc_info=True)

# --- End Token Rate Limiting Helper Functions ---

def setup_api_keys() -> None:
    """Set up API keys from environment variables."""
    providers = ['OPENAI', 'ANTHROPIC', 'GROQ', 'OPENROUTER']
    for provider in providers:
        key = getattr(config, f'{provider}_API_KEY')
        if key:
            logger.debug(f"API key set for provider: {provider}")
        else:
            logger.warning(f"No API key found for provider: {provider}")

    # Set up OpenRouter API base if not already set
    if config.OPENROUTER_API_KEY and config.OPENROUTER_API_BASE:
        os.environ['OPENROUTER_API_BASE'] = config.OPENROUTER_API_BASE
        logger.debug(f"Set OPENROUTER_API_BASE to {config.OPENROUTER_API_BASE}")

    # Set up AWS Bedrock credentials
    aws_access_key = config.AWS_ACCESS_KEY_ID
    aws_secret_key = config.AWS_SECRET_ACCESS_KEY
    aws_region = config.AWS_REGION_NAME

    if aws_access_key and aws_secret_key and aws_region:
        logger.debug(f"AWS credentials set for Bedrock in region: {aws_region}")
        # Configure LiteLLM to use AWS credentials
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        os.environ['AWS_REGION_NAME'] = aws_region
    else:
        logger.warning(f"Missing AWS credentials for Bedrock integration - access_key: {bool(aws_access_key)}, secret_key: {bool(aws_secret_key)}, region: {aws_region}")

async def handle_error(error: Exception, attempt: int, max_attempts: int) -> None:
    """Handle API errors with appropriate delays and logging."""
    delay = RETRY_DELAY  # Default delay

    if isinstance(error, LLMTokenRateLimitError) and error.retry_after is not None:
        delay = error.retry_after
        logger.warning(
            f"Internal token rate limit hit on attempt {attempt + 1}/{max_attempts} for model {error.model_name}. "
            f"Limit: {error.limit} TPM, Usage: {error.usage}. Waiting {delay}s."
        )
    elif isinstance(error, litellm.exceptions.RateLimitError):
        delay = RATE_LIMIT_DELAY  # Specific delay for external rate limits
        logger.warning(f"External rate limit hit on attempt {attempt + 1}/{max_attempts}: {str(error)}")
    else:
        logger.warning(f"Error on attempt {attempt + 1}/{max_attempts}: {str(error)}")

    logger.debug(f"Waiting {delay} seconds before retry...")
    await asyncio.sleep(delay)

def prepare_params(
    messages: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    response_format: Optional[Any] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low'
) -> Dict[str, Any]:
    """Prepare parameters for the API call."""
    params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format,
        "top_p": top_p,
        "stream": stream,
    }

    if api_key:
        params["api_key"] = api_key
    if api_base:
        params["api_base"] = api_base
    if model_id:
        params["model_id"] = model_id

    # Handle token limits
    if max_tokens is not None:
        # For Claude 3.7 in Bedrock, do not set max_tokens or max_tokens_to_sample
        # as it causes errors with inference profiles
        if model_name.startswith("bedrock/") and "claude-3-7" in model_name:
            logger.debug(f"Skipping max_tokens for Claude 3.7 model: {model_name}")
            # Do not add any max_tokens parameter for Claude 3.7
        else:
            param_name = "max_completion_tokens" if 'o1' in model_name else "max_tokens"
            params[param_name] = max_tokens

    # Add tools if provided
    if tools:
        params.update({
            "tools": tools,
            "tool_choice": tool_choice
        })
        logger.debug(f"Added {len(tools)} tools to API parameters")

    # # Add Claude-specific headers
    if "claude" in model_name.lower() or "anthropic" in model_name.lower():
        params["extra_headers"] = {
            # "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"
            "anthropic-beta": "output-128k-2025-02-19"
        }
        logger.debug("Added Claude-specific headers")

    # Add OpenRouter-specific parameters
    if model_name.startswith("openrouter/"):
        logger.debug(f"Preparing OpenRouter parameters for model: {model_name}")

        # Add optional site URL and app name from config
        site_url = config.OR_SITE_URL
        app_name = config.OR_APP_NAME
        if site_url or app_name:
            extra_headers = params.get("extra_headers", {})
            if site_url:
                extra_headers["HTTP-Referer"] = site_url
            if app_name:
                extra_headers["X-Title"] = app_name
            params["extra_headers"] = extra_headers
            logger.debug(f"Added OpenRouter site URL and app name to headers")

    # Add Bedrock-specific parameters
    if model_name.startswith("bedrock/"):
        logger.debug(f"Preparing AWS Bedrock parameters for model: {model_name}")

        if not model_id and "anthropic.claude-3-7-sonnet" in model_name:
            params["model_id"] = "arn:aws:bedrock:us-west-2:935064898258:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            logger.debug(f"Auto-set model_id for Claude 3.7 Sonnet: {params['model_id']}")

    # Apply Anthropic prompt caching (minimal implementation)
    # Check model name *after* potential modifications (like adding bedrock/ prefix)
    effective_model_name = params.get("model", model_name) # Use model from params if set, else original
    if "claude" in effective_model_name.lower() or "anthropic" in effective_model_name.lower():
        messages = params["messages"] # Direct reference, modification affects params

        # Ensure messages is a list
        if not isinstance(messages, list):
            return params # Return early if messages format is unexpected

        # 1. Process the first message if it's a system prompt with string content
        if messages and messages[0].get("role") == "system":
            content = messages[0].get("content")
            if isinstance(content, str):
                # Wrap the string content in the required list structure
                messages[0]["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(content, list):
                 # If content is already a list, check if the first text block needs cache_control
                 for item in content:
                     if isinstance(item, dict) and item.get("type") == "text":
                         if "cache_control" not in item:
                             item["cache_control"] = {"type": "ephemeral"}
                             break # Apply to the first text block only for system prompt

        # 2. Find and process relevant user and assistant messages
        last_user_idx = -1
        second_last_user_idx = -1
        last_assistant_idx = -1

        for i in range(len(messages) - 1, -1, -1):
            role = messages[i].get("role")
            if role == "user":
                if last_user_idx == -1:
                    last_user_idx = i
                elif second_last_user_idx == -1:
                    second_last_user_idx = i
            elif role == "assistant":
                if last_assistant_idx == -1:
                    last_assistant_idx = i

            # Stop searching if we've found all needed messages
            if last_user_idx != -1 and second_last_user_idx != -1 and last_assistant_idx != -1:
                 break

        # Helper function to apply cache control
        def apply_cache_control(message_idx: int, message_role: str):
            if message_idx == -1:
                return

            message = messages[message_idx]
            content = message.get("content")

            if isinstance(content, str):
                message["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        if "cache_control" not in item:
                           item["cache_control"] = {"type": "ephemeral"}

        # Apply cache control to the identified messages
        apply_cache_control(last_user_idx, "last user")
        apply_cache_control(second_last_user_idx, "second last user")
        apply_cache_control(last_assistant_idx, "last assistant")

    # Add reasoning_effort for Anthropic models if enabled
    use_thinking = enable_thinking if enable_thinking is not None else False
    is_anthropic = "anthropic" in effective_model_name.lower() or "claude" in effective_model_name.lower()

    if is_anthropic and use_thinking:
        effort_level = reasoning_effort if reasoning_effort else 'low'
        params["reasoning_effort"] = effort_level
        params["temperature"] = 1.0 # Required by Anthropic when reasoning_effort is used
        logger.info(f"Anthropic thinking enabled with reasoning_effort='{effort_level}'")

    return params

async def make_llm_api_call(
    messages: List[Dict[str, Any]],
    model_name: str,
    # account_id is not used in prepare_params, but make_llm_api_call needs it
    # account_id: Optional[str] = None, 
    response_format: Optional[Any] = None,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    stream: bool = False,
    top_p: Optional[float] = None,
    model_id: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = 'low',
    account_id: Optional[str] = None  # Added account_id
) -> Union[Dict[str, Any], AsyncGenerator]:
    """
    Make an API call to a language model using LiteLLM.

    Args:
        messages: List of message dictionaries for the conversation
        model_name: Name of the model to use (e.g., "gpt-4", "claude-3", "openrouter/openai/gpt-4", "bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
        # account_id: ID of the account making the call, for rate limiting
        response_format: Desired format for the response
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in the response
        tools: List of tool definitions for function calling
        tool_choice: How to select tools ("auto" or "none")
        api_key: Override default API key
        api_base: Override default API base URL
        stream: Whether to stream the response
        top_p: Top-p sampling parameter
        model_id: Optional ARN for Bedrock inference profiles
        enable_thinking: Whether to enable thinking
        reasoning_effort: Level of reasoning effort
        account_id: Optional ID of the account making the call for rate limiting.

    Returns:
        Union[Dict[str, Any], AsyncGenerator]: API response or stream

    Raises:
        LLMRetryError: If API call fails after retries
        LLMError: For other API-related errors
    """
    # debug <timestamp>.json messages
    logger.info(f"Making LLM API call to model: {model_name} (Thinking: {enable_thinking}, Effort: {reasoning_effort}, Account: {account_id})")
    
    # Pre-call token rate limiting check
    if account_id and config.USER_SPECIFIC_TOKEN_RATE_LIMITS_ENABLED:
        try:
            estimated_input_tokens = 0
            if messages: # Ensure messages is not None or empty
                # Ensure litellm.token_counter is available and messages are in correct format
                # This might need more robust error handling if messages structure varies
                try:
                    estimated_input_tokens = litellm.token_counter(model=model_name, messages=messages)
                except Exception as e:
                    logger.warning(f"Could not estimate input tokens for rate limiting pre-check: {e}. Proceeding without pre-check.")
            
            if estimated_input_tokens > 0: # Only check if we have an estimate
                logger.debug(f"Performing pre-call token rate limit check for account {account_id}, model {model_name}, estimated tokens {estimated_input_tokens}.")
                await check_token_rate_limit(account_id, model_name, estimated_input_tokens)
            else:
                logger.debug(f"Skipping pre-call token rate limit check for account {account_id}, model {model_name} due to zero or unestimated input tokens.")

        except LLMTokenRateLimitError as e:
            logger.warning(f"Pre-call token rate limit failed for account {account_id}, model {model_name}: {e.message}")
            raise # Re-raise to be caught by the main error handling or caller

    logger.info(f"📡 API Call: Using model {model_name}")
    params = prepare_params(
        messages=messages,
        model_name=model_name,
        # account_id=account_id, # Not needed by prepare_params itself
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        api_key=api_key,
        api_base=api_base,
        stream=stream,
        top_p=top_p,
        model_id=model_id,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort
    )
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Attempt {attempt + 1}/{MAX_RETRIES} for model {model_name}, account {account_id}")
            # logger.debug(f"API request parameters: {json.dumps(params, indent=2)}")

            response = await litellm.acompletion(**params)
            logger.debug(f"Successfully received API response from {model_name} for account {account_id}")
            # logger.debug(f"Response: {response}") # Can be very verbose

            # Post-call token usage update
            if account_id and not stream and hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                if prompt_tokens is not None and completion_tokens is not None:
                    logger.debug(f"Attempting to update token usage for account {account_id}, model {model_name}. Prompt: {prompt_tokens}, Completion: {completion_tokens}")
                    await update_token_usage(account_id, model_name, prompt_tokens, completion_tokens)
                else:
                    logger.debug(f"Token usage attribute found but prompt/completion tokens missing for account {account_id}, model {model_name}.")
            elif account_id and stream:
                logger.debug(f"Token usage update skipped for streaming response for account {account_id}, model {model_name}.")
            
            return response

        except LLMTokenRateLimitError as e:
            last_error = e
            logger.warning(f"Internal token rate limit hit during attempt {attempt + 1}/{MAX_RETRIES} for account {account_id}, model {model_name}: {e.message}")
            if attempt == MAX_RETRIES - 1:
                error_msg = f"Failed after {MAX_RETRIES} attempts due to internal token rate limit for account {account_id}, model {model_name}. Last error: {str(last_error)}"
                logger.error(error_msg)
                raise LLMRetryError(error_msg) from e
            await handle_error(e, attempt, MAX_RETRIES)

        except (litellm.exceptions.RateLimitError, OpenAIError, json.JSONDecodeError) as e: # Keep OpenAIError and JSONDecodeError if they are still relevant
            last_error = e
            # handle_error now differentiates external RateLimitError
            if attempt == MAX_RETRIES - 1:
                error_msg = f"Failed after {MAX_RETRIES} attempts for account {account_id}, model {model_name}. Last error: {str(last_error)}"
                logger.error(error_msg)
                raise LLMRetryError(error_msg) from e
            await handle_error(e, attempt, MAX_RETRIES)
            
        except Exception as e:
            logger.error(f"Unexpected error during API call for account {account_id}, model {model_name}: {str(e)}", exc_info=True)
            # This could be a litellm.exceptions.APIConnectionError, BadRequestError, AuthenticationError, etc.
            # Or a non-LiteLLM error.
            raise LLMError(f"API call failed for account {account_id}, model {model_name}: {str(e)}") from e

    # This part should ideally not be reached if all attempts lead to exceptions handled above.
    # However, as a fallback:
    final_error_msg = f"Failed to make API call after {MAX_RETRIES} attempts for account {account_id}, model {model_name}"
    if last_error:
        final_error_msg += f". Last error: {str(last_error)}"
    logger.error(final_error_msg, exc_info=True if last_error else False)
    raise LLMRetryError(final_error_msg)

# Initialize API keys on module import
setup_api_keys()

# Test code for OpenRouter integration
async def test_openrouter(test_account_id: Optional[str] = "test_user_or_account"):
    """Test the OpenRouter integration with a simple query."""
    logger.info(f"--- Testing OpenRouter with Account ID: {test_account_id} ---")
    test_messages = [
        {"role": "user", "content": "Hello, can you give me a quick test response?"}
    ]

    try:
        # Test with standard OpenRouter model
        print("\n--- Testing standard OpenRouter model ---")
        response = await make_llm_api_call(
            model_name="openrouter/openai/gpt-4o-mini",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100,
            account_id=test_account_id
        )
        print(f"Response: {response.choices[0].message.content}")

        # Test with deepseek model
        print("\n--- Testing deepseek model ---")
        response = await make_llm_api_call(
            model_name="openrouter/deepseek/deepseek-r1-distill-llama-70b",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100,
            account_id=test_account_id
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")

        # Test with Mistral model
        print("\n--- Testing Mistral model ---")
        response = await make_llm_api_call(
            model_name="openrouter/mistralai/mixtral-8x7b-instruct",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100,
            account_id=test_account_id
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")

        return True
    except Exception as e:
        print(f"Error testing OpenRouter: {str(e)}")
        return False

async def test_bedrock(test_account_id: Optional[str] = "test_user_or_account"):
    """Test the AWS Bedrock integration with a simple query."""
    logger.info(f"--- Testing Bedrock with Account ID: {test_account_id} ---")
    test_messages = [
        {"role": "user", "content": "Hello, can you give me a quick test response?"}
    ]

    try:
        response = await make_llm_api_call(
            model_name="bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
            model_id="arn:aws:bedrock:us-west-2:935064898258:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            messages=test_messages,
            temperature=0.7,
            account_id=test_account_id
            # Claude 3.7 has issues with max_tokens, so omit it
            # max_tokens=100
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model used: {response.model}")

        return True
    except Exception as e:
        print(f"Error testing Bedrock: {str(e)}")
        return False

if __name__ == "__main__":
    import asyncio
    
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG) # Ensure our logger is also at debug for testing verbosity

    test_account = "dev_tester_001"
    
    # Test Bedrock
    print("\n--- Running Bedrock Test ---")
    test_success_br = asyncio.run(test_bedrock(test_account_id=test_account))
    if test_success_br:
        print("\n✅ Bedrock integration test completed successfully!")
    else:
        print("\n❌ Bedrock integration test failed!")

    # Test OpenRouter
    # print("\n--- Running OpenRouter Test ---")
    # test_success_or = asyncio.run(test_openrouter(test_account_id=test_account))
    # if test_success_or:
    #     print("\n✅ OpenRouter integration test completed successfully!")
    # else:
    #     print("\n❌ OpenRouter integration test failed!")
    
    # Note: To fully test OpenRouter, ensure OPENROUTER_API_KEY is set in your environment or .env file.
    # If you want to run both, uncomment the OpenRouter test block.
    # For now, focusing on Bedrock as per the original single test.
    # To run both, you might do:
    # test_success = test_success_br and test_success_or (if OpenRouter is uncommented)
    # For now, just use Bedrock's success for the final message.
    test_success = test_success_br

    if test_success: # This will be true if Bedrock test was successful
        print("\n✅ All enabled integration tests completed successfully!")
    else:
        print("\n❌ Some integration tests failed!")
