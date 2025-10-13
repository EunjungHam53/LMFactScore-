from pathlib import Path
import json
import argparse
import os

from pysbd import Segmenter
from tiktoken import Encoding

from .knowledge_graph import PROMPT_FILE_PATH
from .openai_api import (RESPONSES_DIRECTORY_PATH,
                        get_max_chapter_segment_token_count,
                        get_openai_model_encoding, save_openai_api_response)
from .utils import (execute_function_in_parallel, set_up_logging,
                   strip_and_remove_empty_strings)

logger = set_up_logging('openai-api-scripts.log')


def get_paragraphs(text):
    """Split a text into paragraphs."""
    paragraphs = strip_and_remove_empty_strings(text.split('\n\n'))
    paragraphs = [' '.join(paragraph.split()) for paragraph in paragraphs]
    return paragraphs


def combine_text_subunits_into_segments(subunits, join_string,
                                        encoding: Encoding,
                                        max_token_count):
    """Combine subunits of text into segments that do not exceed a maximum number of tokens."""
    subunit_token_counts = [len(tokens) for tokens
                            in encoding.encode_ordinary_batch(subunits)]
    join_string_token_count = len(encoding.encode_ordinary(join_string))
    total_token_count = (sum(subunit_token_counts) + join_string_token_count
                         * (len(subunits) - 1))
    if total_token_count <= max_token_count:
        return [join_string.join(subunits)]
    # Calculate the approximate number of segments and the approximate number of tokens per segment, in order to keep the segment lengths roughly equal.
    approximate_segment_count = total_token_count // max_token_count + 1
    approximate_segment_token_count = round(total_token_count
                                            / approximate_segment_count)
    segments = []
    current_segment_subunits = []
    current_segment_token_count = 0
    for i, (subunit, subunit_token_count) in enumerate(
            zip(subunits, subunit_token_counts)):
        # The token count if the current subunit is added to the current segment.
        extended_segment_token_count = (current_segment_token_count
                                        + join_string_token_count
                                        + subunit_token_count)
        # Add the current subunit to the current segment if it results in a
        # token count that is closer to the approximate segment token count
        # than the current segment token count.
        if (extended_segment_token_count <= max_token_count
                and abs(extended_segment_token_count
                        - approximate_segment_token_count)
                <= abs(current_segment_token_count
                       - approximate_segment_token_count)):
            current_segment_subunits.append(subunit)
            current_segment_token_count = extended_segment_token_count
        else:
            segment = join_string.join(current_segment_subunits)
            segments.append(segment)
            # If it is possible to join the remaining subunits into a single
            # segment, do so. Additionally, add the current subunit as a
            # segment if it is the last subunit.
            if (sum(subunit_token_counts[i:]) + join_string_token_count
                    * (len(subunits) - i - 1) <= max_token_count
                    or i == len(subunits) - 1):
                segment = join_string.join(subunits[i:])
                segments.append(segment)
                break
            current_segment_subunits = [subunit]
            current_segment_token_count = subunit_token_count
    return segments


def split_long_sentences(sentences, encoding: Encoding,
                         max_token_count):
    """Given a list of sentences, split sentences that exceed a maximum number of tokens into multiple segments."""
    token_counts = [len(tokens) for tokens
                    in encoding.encode_ordinary_batch(sentences)]
    split_sentences = []
    for sentence, token_count in zip(sentences, token_counts):
        if token_count > max_token_count:
            words = sentence.split()
            segments = combine_text_subunits_into_segments(
                words, ' ', encoding, max_token_count)
            split_sentences.extend(segments)
        else:
            split_sentences.append(sentence)
    return split_sentences


def split_long_paragraphs(paragraphs, encoding: Encoding,
                          max_token_count):
    """Given a list of paragraphs, split paragraphs that exceed a maximum number of tokens into multiple segments."""
    token_counts = [len(tokens) for tokens
                    in encoding.encode_ordinary_batch(paragraphs)]
    split_paragraphs = []
    for paragraph, token_count in zip(paragraphs, token_counts):
        if token_count > max_token_count:
            sentences = Segmenter().segment(paragraph)
            sentences = split_long_sentences(sentences, encoding,
                                             max_token_count)
            segments = combine_text_subunits_into_segments(
                sentences, ' ', encoding, max_token_count)
            split_paragraphs.extend(segments)
        else:
            split_paragraphs.append(paragraph)
    return split_paragraphs


def get_chapter_segments(chapter_text, encoding: Encoding,
                         max_token_count):
    """Split a chapter text into segments that do not exceed a maximum number of tokens."""
    paragraphs = get_paragraphs(chapter_text)
    paragraphs = split_long_paragraphs(paragraphs, encoding, max_token_count)
    chapter_segments = combine_text_subunits_into_segments(
        paragraphs, '\n', encoding, max_token_count)
    return chapter_segments


def get_response_save_path(idx, save_path, project_gutenberg_id,
                           chapter_index = None,
                           chapter_segment_index = None,
                           chapter_segment_count = None):
    """Get the path to the JSON file(s) containing response data from the OpenAI API."""
    save_path = Path(save_path)
    os.makedirs(save_path, exist_ok=True)

    if chapter_index is not None:
        save_path /= str(chapter_index)
        if chapter_segment_index is not None:
            save_path /= (f'{chapter_segment_index + 1}-of-'
                          f'{chapter_segment_count}.json')
    return save_path


def save_openai_api_responses_for_script(script, prompt, encoding, max_chapter_segment_token_count, idx, save_path):
    """Call the OpenAI API for each chapter segment in a script and save the responses to JSON files."""
    project_gutenberg_id = script['id']
    chapter_count = len(script['chapters'])
    logger.info(f'Starting to call OpenAI API and save responses for script '
                f'{project_gutenberg_id} ({chapter_count} chapters).')
    prompt_message_lists = []
    save_paths = []
    for chapter in script['chapters']:
        chapter_index = chapter['index']
        chapter_responses_directory = get_response_save_path(idx, save_path, project_gutenberg_id, chapter_index)
        chapter_response_paths = list(chapter_responses_directory.glob('*.json'))
        if chapter_response_paths:
            chapter_segment_count = int(chapter_response_paths[0].stem.split('-')[-1])
            if len(chapter_response_paths) == chapter_segment_count:
                logger.info(f'Skipping already saved response(s) for chapter '
                            f'{chapter_index}.')
                continue
        chapter_segments = chapter['text']
        chapter_segment_count = len(chapter_segments)
        for chapter_segment_index, chapter_segment in enumerate(chapter_segments):
            prompt_with_story = prompt.replace('{STORY}', chapter_segment)
            response_save_path = get_response_save_path(
                idx, save_path, project_gutenberg_id, chapter_index, chapter_segment_index,
                chapter_segment_count)
            prompt_message_lists.append([{'role': 'user',
                                          'content': prompt_with_story}])
            save_paths.append(response_save_path)
    execute_function_in_parallel(save_openai_api_response,
                                 prompt_message_lists, save_paths)
    logger.info(f'Finished saving responses for script {project_gutenberg_id}.')


def save_triples_for_scripts(input_data, idx, save_path):
    """Call the OpenAI API to generate knowledge graph nodes and edges, and save the responses to JSON files."""
    
    # 1) load data
    script = input_data

    # 2) call OpenAI API
    prompt = PROMPT_FILE_PATH.read_text()
    max_chapter_segment_token_count = get_max_chapter_segment_token_count(prompt)
    encoding = get_openai_model_encoding()
    save_openai_api_responses_for_script(script, prompt, encoding, max_chapter_segment_token_count, idx, save_path)
