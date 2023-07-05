import openai
import json
import re
from collections import Counter
from typing import List, Optional, Union

openai.api_key = 'sk-<OPENAI_API_KEY>'

"""
https://platform.openai.com/docs/models/model-endpoint-compatibility
"""
AVAILABLE_MODELS = [
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "text-davinci-003",
    "code-davinci-001",
]

"""
prompts
"""
prompt_001 = """\
Your task is to first think through it step by step, and then provide the rationale and your option.
"""
prompt_002 = """\
Let's first read and understand the problem carefully, extract relevant variables and their corresponding numerals, \
and make a complete plan. Then, let's carry out the plan, calculate intermediate variables \
(pay attention to correct numerical calculation and commonsense), \
solve the problem step by step, and show the answer.
"""
prompt_003 = """\
Let's first understand the problem, extract relevant variables and their corresponding numerals, \
and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables \
(pay attention to correct numerical calculation and commonsense), \
solve the problem step by step, and show the answer.
"""
prompt_004 = """\
Let's first understand the problem (please take care to extract the mathematical relationships implicit \
in the textual descriptions),  extract relevant variables and their corresponding numerals,\
and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables \
(pay attention to correct numerical calculation and commonsense), \
solve the problem step by step, and show the answer.
"""
prompt_005 = """\
Let's solve the problem step by step:
First: 1. understand the problem(please take care to extract the mathematical relationships implicit in the textual descriptions);\
 2. extract relevant variables and their corresponding numerals\
 3. make and devise a complete plan.
Then: 4. let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), \
solve the problem step by step, and show the answer.
"""

"""
format prompts
"""
format_prompt_001 = """\
Use the following format:
Answer: <your option>
"""
format_prompt_002 = """\
Use the following format:
Rationale: <Detailed solution steps>
Answer: <your option>
"""
format_prompt_003 = """\
Use the following format:
Rationale: <Concise and logical steps to solve the problem>
Answer: <must choose and only choose one of the 5 options provided above ```No explanation is needed!```>
"""
format_prompt_004 = """\
Use the following format:
Rationale: <Concise and logical steps to solve the problem>
Answer: <must choose and only choose one of the 5 options provided above \
(if none match, You have to choose an option at random) ```No explanation is needed!```>
"""


def chat_generate_text(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    system_prompt: str = "You are an excellent test taker.",
    temperature: float = 0,
    max_tokens: int = 1024,
    n: int = 1,
    stop: Optional[Union[str, list]] = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
) -> str:
    """
    chat_generate_text - Generates text using the OpenAI API.
    :param str prompt: prompt for the model
    :param str model: model to use, defaults to "gpt-3.5-turbo"
    :param str system_prompt: initial prompt for the model, defaults to "You are an excellent test taker."
    :param float temperature: model temperature, defaults to 0.5
    :param int max_tokens: max_tokens, defaults to 1024
    :param int n: n of text, defaults to 1
    :param Optional[Union[str, list]] stop: stop, defaults to None
    :param float presence_penalty: presence_penalty, defaults to 0
    :param float frequency_penalty: frequency_penalty, defaults to 0.1
    :return str: generated_text
    """
    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )

    generated_text = response.choices[0].message["content"].strip()
    return generated_text


def get_answer(
    response: str
) -> str:
    """
    extract answer(A, B, C...) from model response
    """
    option = re.search(r"\[([A-Z])\)", response)
    extracted_option = ""
    if option:
        extracted_option = option.group(1)
    else:
        print("Option not found.")

    return extracted_option


def gen_zero_shot_prompt(
    question: str, 
    options: str,
) -> str:
    """
    Generate the prompt for zero_shot
    """
    return f"""
Question: {question}
Options: [{', ['.join(options)}
Answer the question and give only your options without any explanation.

{format_prompt_001}
"""


def gen_zero_shot_CoT_prompt(
    question: str, 
    options: str,
) -> str:
    """
    Generate the prompt for zero_shot_CoT
    """
    return f"""
Question: {question}
Options: [{', ['.join(options)}
{prompt_005}

{format_prompt_003}
"""


def gen_self_consistency_prompt(
    question: str, 
    options: str,
) -> str:
    """
    Generate the prompt for self_consistency
    """
    return f"""
Question: {question}
Options: [{', ['.join(options)}
{prompt_005}

{format_prompt_003}
"""


def zero_shot_generate(
    question: str, 
    options: str,
) -> str:
    """
    Return answer by zero_shot method
    """
    prompt = gen_zero_shot_prompt(question, options)
    response = chat_generate_text(prompt)
    answer = get_answer(response)
    return answer


def zero_shot_CoT_generate(
    question: str, 
    options: str,
) -> str:
    """
    Return answer by zero_shot_CoT method
    """
    prompt = gen_zero_shot_CoT_prompt(question, options)
    response = chat_generate_text(prompt)
    answer = get_answer(response)

    return answer


def self_consistency_generate(
    question: str, 
    options: str,
    temperature: float = 0.7,
    num_path: int = 5,
) -> str:
    """
    Return answer by self_consistency method
    """
    answers = []
    for i in range(num_path):
        prompt = gen_self_consistency_prompt(question, options)
        response = chat_generate_text(prompt, temperature)
        answer = get_answer(response)
        answers.append(answer)

    counter = Counter(answers)
    most_common_answer = counter.most_common(1)[0][0]

    return most_common_answer


def main():
    # load dataset
    with open('data/test.json', 'r') as file:
        dataset = json.load(file)
    
    idx = 0 # test exmaple's index
    start_idx = 0
    end_idx = 254
    cnt = 0 # total test number
    correct_cnt = 0 # correct number

    method = "self_consistency"
    output_file = f"result/{method}_result.txt"
    
    for problem in dataset:
        idx = idx + 1 # base 1
        if idx > start_idx and idx <= end_idx: # test it
            cnt = cnt + 1
            print(f"start test {cnt} (index = {idx})")
            question = problem['question']
            options = problem['options']
            correct = problem['correct']

            # change your method here
            answer = self_consistency_generate(question, options)
            
            if answer == correct:
                print("Bingo")
                correct_cnt = correct_cnt + 1
            else:
                pass

            with open(output_file, "a") as file:
                file.write(f"{idx},{answer},{correct}\n")
        else:
            pass

    accuracy_rate = correct_cnt / cnt
    print(f"correct_cnt = {correct_cnt}")
    print(f"cnt = {cnt}")
    print(f"accuracy_rate = {accuracy_rate}")


if __name__ == '__main__':
    main()