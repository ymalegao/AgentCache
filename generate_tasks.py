#from base tasks, #have an LLM take on persona and generate task based on how they would provide/talk to the model. 

#exampe: baese: task = "Write a hello world program"

#persona pertrubed: " Please write me an helllo world program in C++"



#persona directory:
PERSONA_PROMPT = """
You are the person described in the YAML below. You are typing a message to an AI coding agent to get a task done.

<PERSONA_BEHAVIORAL_SPEC>
{persona_profile}
</PERSONA_BEHAVIORAL_SPEC>

## Instructions
Rewrite the base task as a single, natural message from this persona to an AI coding agent.

- **Linguistic Immersion**: Adopt the vocabulary, syntax, sentence length, and grammatical quirks of this persona exactly as defined in the YAML. Pay close attention to the `example_utterances` — match that register.
- **Anti-LLM Formatting**: Do not write like an AI. No formal commands, no "deliver the implementation", no "execute". Write like a real human typing a message.
- **Emotional Expression**: Let the persona's traits dictate punctuation, capitalization, and brevity. An impatient persona is short. A skeptical persona asks questions. An activist uses their vocabulary naturally, not constantly.
- Do not directly copy phrases from the example utterances. Use them only as a baseline for the register and tone.
- Ensure the emotional reaction matches the task. Do not act furiously angry if you are just asking a simple syntax question. Only escalate if the task implies a prior failure."
- If the persona is obsessed with efficiency, the message must be extremely short. Do not add filler complaints or conversational padding.
- Do not add technical requirements not in the base task.
- Output only the message. Nothing else.

Base task: {base_task}
""".strip()


import os
import yaml
from openai import OpenAI
from dotenv import load_dotenv
import random
import json



load_dotenv()

PERSONA_DIR = "personas"
#NAUT_END_POINT="https://ellm.nrp-nautilus.io/v1"


client = OpenAI(
    api_key=os.environ.get("NAUT_API_KEY"),
    base_url="https://ellm.nrp-nautilus.io/v1",
)


TASKS_RAW = [
    # Bug fixes
    "I have a Python function that's supposed to return the factorial of a number but it returns wrong results for 0. Here's the code: def factorial(n): if n == 1: return 1 else: return n * factorial(n-1). Fix it.",
    "My binary search implementation never finds the target element even when it exists in the list. def binary_search(arr, target): low, high = 0, len(arr) while low < high: mid = (low+high)//2 if arr[mid] == target: return mid elif arr[mid] < target: low = mid else: high = mid return -1. What's wrong?",
    "This decorator is supposed to retry a function 3 times on exception but it only tries once. def retry(func): def wrapper(*args): try: return func(*args) except: return func(*args). Fix it.",
    "My CSV parser breaks on lines that have commas inside quoted fields. It's splitting on every comma. Fix the parsing logic.",
    "I have a race condition in my Python threading code. Two threads are incrementing a shared counter and the final value is always less than expected.",

    # Feature requests
    "Add a timeout parameter to this function that makes an HTTP GET request using the requests library. Right now it hangs forever if the server doesn't respond.",
    "I have a function that reads a JSON config file. Add support for environment variable overrides so that any key in the config can be overridden by an env var with the same name uppercased.",
    "Extend this basic logging setup to write to both console and a rotating file that caps at 10MB and keeps 3 backups.",
    "Add input validation to this Flask endpoint that accepts a JSON body with name, email, and age fields. Return proper 400 errors with messages for missing or invalid fields.",
    "I have a script that processes files sequentially. Parallelize it using Python's concurrent.futures so it processes up to 4 files at a time.",

    # Refactoring
    "This function is 200 lines long and does parsing, validation, and database insertion all in one. Break it into smaller functions with single responsibilities.",
    "I have 6 nearly identical functions that each handle a different file format (csv, json, xml, yaml, toml, ini). Refactor them into a single function with a format parameter.",
    "Replace all the raw SQL string concatenation in this file with parameterized queries to prevent SQL injection.",
    "This code uses a global variable to track state between function calls. Refactor it to use a class instead.",

    # Explanation / understanding
    "What is the difference between deepcopy and shallow copy in Python and when would each cause bugs in my code?",
    "My Python script is slow when processing a list of 1 million items. Walk me through how to profile it and identify the bottleneck.",
    "Explain why my generator function is only iterable once and how to make it reusable.",
    "Why does modifying a list inside a function affect the original list outside the function but doing the same with an integer doesn't?",

    # Testing
    "Write unit tests for a stack implementation that has push, pop, peek, and is_empty methods. Cover edge cases.",
    "I have a function that calls an external payment API. Write tests for it that mock the API so the tests don't make real network calls.",
]


def load_personas():
    personas = {}
    for filename in os.listdir(PERSONA_DIR):
        if filename.endswith(".yaml"):
            with open(os.path.join(PERSONA_DIR, filename), "r") as f:
                personas[filename] = yaml.safe_load(f)
    return personas

def perturb_task_with_persona(base_task: str, persona_prompt: str, persona_data: dict, model="qwen3-small") -> str:
    
    pp = yaml.dump(persona_data, default_flow_style=False)
    prompt = persona_prompt.replace("{persona_profile}", pp)
    prompt = prompt.replace("{base_task}", base_task)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.9
    )
    return response

if __name__ == "__main__":
    personas = load_personas()
    final_array = []
    for task in TASKS_RAW:

        #load random persona, petrub task, save to file. 
        random_persona = random.choice(list(personas.keys()))
        random_persona_data = personas[random_persona]

        r = perturb_task_with_persona(task, PERSONA_PROMPT, random_persona_data)
        perturbed_task = r.choices[0].message.content
        final_array.append({"base_task": task, "perturbed_task": perturbed_task})
        
    
    print(final_array)
    with open("tasks.json", "w") as f:
        json.dump(final_array, f)
