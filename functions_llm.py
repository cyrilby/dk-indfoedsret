"""
===================================================
Customs function for extracting features using LLMs
===================================================

Lavet af: kirilboyanovbg[at]gmail.com
Sidste opdatering: 08-11-2024

In this script, we create several different functions that allow
us to extract relevant vessel features using OpenAI's LLM model.
These can then be applied to either Softship data or data
originating from relevant mailboxes.
"""

# %% Setting things up

# Importing relevant packages
import re
import time
import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm
import yaml


# %% Custom function for getting access to the API


def connect_to_openai(access_params: str) -> AzureOpenAI:
    """
    Imports the access key for the OpenAI API from a local TXT file,
    then creates and returns a client object.

    Args:
        access_params (str): file path to TXT access key and other
        relevant info such as endpoint, API version, etc.

    Returns:
        OpenAI: OpenAI client object.
    """

    # Opening YAML file containing API key & parameters
    with open(access_params, "r") as file:
        access_params = yaml.safe_load(file)

    # Extracting the relevant parameters from the YAML file
    api_key = access_params["openai.api_key"]
    api_version = access_params["openai.api_version"]
    azure_endpoint = access_params["openai.azure_endpoint"]

    # Initiating OpenAI client
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version
    )
    return client


# %% Custom function for loading prompt from TXT file


def get_prompt(prompt_file: str) -> str:
    """
    Opens a TXT file containing a prompt for the OpenAI LLM.
    Extract the contents of the actual prompt, which are
    enclosed between <prompt> and </prompt>.

    Args:
        prompt_file (str): path to TXT file

    Returns:
        str: prompt text for use
    """
    # Importing prompt from TXT file
    with open(f"prompts/{prompt_file}", "r", encoding="utf-8") as file:
        prompt_full_text = file.read()

    # Regular expression to capture content between <prompt> and </prompt>
    match = re.search(r"<prompt>(.*?)</prompt>", prompt_full_text, re.DOTALL)
    if match:
        # Strip leading/trailing whitespace and newlines
        return match.group(1).strip()
    return None


# %% Custom function for a one-time query to the OpenAI model


def query_llm(
    openai_client: AzureOpenAI,
    system_prompt: str,
    user_prompt: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> list | None:
    """
    Queries OpenAI's LLM based on a given system and user prompts.
    Automatically attaches the relevant bit of actual data to the
    user-defined prompt. Returns the response stripped from model
    metadata.

    Args:
        openai_client (OpenAI): OpenAI client object
        system_prompt (str): text to use as system prompt, defining
        the actual question we're asking
        user_prompt (str): text to use as the "data point"
        model_name (str): name of the OpenAI model to use.
        Defaults to "gpt-4o-mini".
        temperature (float, optional): temperature. Pr. definition 0.7.
        max_tokens (int, optional): max_tokens. Pr. definition None.
        content_only (bool, optional): om hele outputtet skal leveres
        eller om kun teksten (indholdet) og antal brugte tokens skal leveres.
        Pr. definition True (det sidstnævnte).

    Returns:
        list|None: list of relevant responses (if found) or None
    """

    # Querying the LLM and recording its response
    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if completion is not None:
        response = completion.choices[0].message.content
        return response
    else:
        return None


# %% Custom function to limit API requests per minute


def query_llm_multiple(
    openai_client: AzureOpenAI,
    system_prompt: str,
    data_points: list[str],
    min_len: int = 3,
    max_rpm: int | None = 20,
    **kwargs,
) -> list:
    """
    Iterates the query_llm() function over a list of different
    values, generating and recording responses to the relevant
    question. Ensures that we do not send more requests per
    minute (RPM) than the API allows for, unless max_rpm is None.

    Args:
        openai_client (OpenAI): OpenAI client object
        system_prompt (str): text to use as system prompt, defining
        the actual question we're asking
        data_points (list[str]): list of text-based data points to
        iterate over.
        min_len (int): min length of the string containing the data,
        if the string is considered valid to be processed by an LLM.
        max_rpm (int, optional): max requests per minute for the
        API. Defaults to 20. Set to None for unlimited requests.
        **kwargs: relevant arguments to be passed on to the
        query_llm() function.

    Returns:
        list: all responses
    """

    # Initializing list as storage and counter for RPM
    all_responses = []
    request_count = 0
    start_time = time.time()

    print("Generating responses using LLM....")
    for text_input in tqdm(data_points, total=len(data_points)):
        # Check if there is any relevant text to use
        if not pd.isnull(text_input):
            text_clean = text_input.strip()
            if len(text_clean) > min_len:
                valid_input = True
            else:
                valid_input = False
        else:
            valid_input = False

        if valid_input:
            # Only apply rate limiting if max_rpm is specified (not None)
            if max_rpm is not None and request_count >= max_rpm:
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    # Wait for the remainder of the minute before continuing
                    time.sleep(60 - elapsed_time)
                # Reset the counter and start time after waiting
                request_count = 0
                start_time = time.time()

            # Query the LLM and increase the request count
            response = query_llm(
                openai_client=openai_client,
                system_prompt=system_prompt,
                user_prompt=text_clean,
                **kwargs,
            )
            request_count += 1
            all_responses.append(response)
        else:
            all_responses.append(None)

    print("Generating responses complete.")
    return all_responses


# %%
