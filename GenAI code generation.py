import json
requirements = {
    "object_detection": {
        "method": "AI/ML",
        "accuracy": "95%",
        "detection_range": "50 meters"
    },
    "collision_warning": {
        "trigger_distance": "5 meters",
        "response_time": "100 ms"
    },
    "compliance": {
        "standards": ["MISRA", "ASPICE", "ISO 26262"]
    },
    "test_coverage": "100%"
}
import openai

with open("fcw_requirements.json", "w") as file:
    json.dump(requirements, file, indent=4)

def generate_code_from_requirements(requirements_file):
    with open(requirements_file, "r") as file:
        requirements = json.load(file)

def generate_prompt(requirements):
    prompt = (
        "Generate Python code for the following requirements:\n"
        + json.dumps(requirements, indent=4)
        + "\nThe code should include object detection, collision warning, and display of annotated video frames."
    )
    return prompt

def generate_code(prompt):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    )
    return response['choices'][0]['message']['content']

openai.api_key = "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"

prompt = generate_prompt(requirements)
generated_code = generate_code(prompt)

with open("generated_code_FCW.py", "w") as code_file:
    code_file.write(generated_code)

print("Generated code saved to generated_code_FCW.py")