import boto3
import json

prompt_data = """
Act as a Shakespeare and write a poem on Genertaive AI
"""

bedrock = boto3.client(service_name="bedrock-runtime")

# for llama2
payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

body = json.dumps(payload)

model_id = "meta.llama-70b-chat-v1"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get('body').read())
response_text = response_body['generation']
print(response_text)
