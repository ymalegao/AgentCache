
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from datasets import load_dataset
load_dotenv()


def generate_good_example():

    client = OpenAI(
        api_key=os.environ.get("NAUT_API_KEY"),
        base_url="https://ellm.nrp-nautilus.io/v1",
    )


    SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer.

    Please solve the issue provided by the user. You can execute bash commands and edit files to implement the necessary changes.

    ## Recommended Workflow
    1. Analyze the codebase by finding and reading relevant files
    2. Create a script to reproduce the issue
    3. Edit the source code to resolve the issue
    4. Verify your fix works by running your script again
    5. Test edge cases to ensure your fix is robust"""




    with open("tasks.json", "r") as f:
        tasks_data = json.load(f)
    TASKS = [item["perturbed_task"] for item in tasks_data]

    OUTPUT_DIR = "good_examples"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, "good_examples.jsonl")

    existing_data = []
    task_to_index = {}

    # Load existing records, track which tasks already have good_examples
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                record = json.loads(line)
                existing_data.append(record)
                task_to_index[record.get("task")] = idx

    print(f"Loaded {len(existing_data)} existing records")

    # Ensure existing_data has an entry for every task, with placeholder if missing
    for i, task in enumerate(TASKS):
        if task not in task_to_index:
            record = {
                "index": i,
                "task": task,
                "good_example": None,
            }
            task_to_index[task] = len(existing_data)
            existing_data.append(record)

    # Regenerate only those with null good_example and update in-place
    null_tasks = [r["task"] for r in existing_data if r.get("good_example") is None]
    print(f"Tasks with null good_example: {len(null_tasks)}")

    for i, task in enumerate(TASKS):
        rec_idx = task_to_index[task]
        record = existing_data[rec_idx]
        if record.get("good_example") is not None:
            continue

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{task}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        response = client.chat.completions.create(
            model="gpt-oss",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.9
        )
        good_example = response.choices[0].message.content
        record["index"] = i
        record["good_example"] = good_example
        existing_data[rec_idx] = record
        print(f"Filled null good_example for task {i+1} of {len(TASKS)}")

    # Rewrite file with updated records (nulls now filled)
    with open(output_path, "w") as out_f:
        for record in existing_data:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    #generate_good_example()
    print("Loading and tokenizing dataset...")
    with open("good_examples/good_examples.jsonl", "r") as f:
        good_examples = [line.strip() for line in f]
    good_examples = [json.loads(item) for item in good_examples]
    print(good_examples)
    dataset = load_dataset("json", data_files={"train": good_examples})
   