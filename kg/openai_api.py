# import json
# import logging
# import os
# from pathlib import Path

# import openai
# from dotenv import load_dotenv
# from openai.error import (APIError, RateLimitError, ServiceUnavailableError,
#                           Timeout, APIConnectionError, InvalidRequestError)
# from tenacity import (before_sleep_log, retry, retry_if_exception_type,
#                       stop_after_delay, wait_random_exponential, stop_after_attempt)
# from tiktoken import Encoding, encoding_for_model

# logger = logging.getLogger(__name__)

# load_dotenv()
# openai.api_key = os.getenv('OPENAI_API_KEY')

# OPENAI_MODEL_ID = 'gpt-4o-mini-2024-07-18'
# # This value is set by OpenAI for the selected model and cannot be changed.
# MAX_MODEL_TOKEN_COUNT = 4096
# # This value can be changed.
# MAX_RESPONSE_TOKEN_COUNT = 512
# RESPONSES_DIRECTORY_PATH = Path('../openai-api-responses-new')


# def get_openai_model_encoding():
#     """Get the encoding (tokenizer) for the OpenAI model."""
#     return encoding_for_model(OPENAI_MODEL_ID)


# def get_max_chapter_segment_token_count(prompt: str) -> int:
#     """
#     Calculate the maximum number of tokens that a chapter segment may contain
#     given the prompt.
#     """
#     encoding = get_openai_model_encoding()
#     # `encode_ordinary()` ignores special tokens and is slightly faster than `encode()`.
#     prompt_token_count = len(encoding.encode_ordinary(prompt))
#     # Subtract 8 for tokens added by OpenAI in the prompt and response (refer to https://platform.openai.com/docs/guides/chat/managing-tokens for details).
#     # Subtract 1 for the newline added below to the end of the prompt.
#     # This calculation does not have to be exact.
#     max_chapter_segment_token_count = (MAX_MODEL_TOKEN_COUNT
#                                        - MAX_RESPONSE_TOKEN_COUNT
#                                        - prompt_token_count - 8 - 1)
#     return max_chapter_segment_token_count


# @retry(retry=retry_if_exception_type((APIError, Timeout, RateLimitError,
#                                       ServiceUnavailableError, APIConnectionError, InvalidRequestError)),
#        wait=wait_random_exponential(max=60), stop=stop_after_attempt(10),
#        before_sleep=before_sleep_log(logger, logging.WARNING))
# def save_openai_api_response(prompt_messages,
#                              save_path):
#     """Use a prompt to make a request to the OpenAI API and save the response to a JSON file."""
#     try:
#         logger.info(f'Calling OpenAI API and saving response to {save_path}...')
#         response = openai.ChatCompletion.create(
#             model=OPENAI_MODEL_ID, messages=prompt_messages, temperature=0)
#         finish_reason = response.choices[0].finish_reason
#         if finish_reason != 'stop':
#             logger.error(f'`finish_reason` is `{finish_reason}` for {save_path}.')
#         save_data = {'model': response.model, 'usage': response.usage,
#                      'finish_reason': finish_reason,
#                      'prompt_messages': prompt_messages,
#                      'response': response.choices[0].message.content}
#     except InvalidRequestError:
#         logger.error(f'InvalidRequestError encountered 10 times. Returning empty string for {save_path}.')
#         save_data = {'model': None, 'usage': None,
#                      'finish_reason': 'invalid_request',
#                      'prompt_messages': prompt_messages,
#                      'response': ' '}
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(save_path, 'w') as save_file:
#         json.dump(save_data, save_file, indent=4, ensure_ascii=False)


# def load_response_text(save_path):
#     """Load the response text from a JSON file containing response data from the OpenAI API."""
#     with open(save_path, 'r') as save_file:
#         save_data = json.load(save_file)
#     return save_data['response']
