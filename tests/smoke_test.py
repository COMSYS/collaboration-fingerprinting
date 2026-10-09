import os
import sys
import json
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mdagents_sidechannel.instrumentation import Config

# Raw connectivity check: talks to the backend directly, bypassing the
# instrumented client, to confirm credentials and endpoint before a real run.
client = OpenAI(api_key=os.environ[Config.api_key_env], base_url=Config.base_url)

message = "Reply with exactly one word: OK"
response = client.chat.completions.create(
    model=Config.model,
    messages=[{"role": "user", "content": message}],
)

choice = response.choices[0]
content = choice.message.content

print("=== visible content ===")
print(repr(content))
print("visible content length (chars):", len(content or ""))

print("\n=== usage ===")
print(response.usage.model_dump_json(indent=2) if response.usage else "MISSING — no usage block returned")

print("\n=== full message object (checking for hidden reasoning fields) ===")
print(choice.message.model_dump_json(indent=2))

print("\n=== full raw response ===")
print(response.model_dump_json(indent=2))
