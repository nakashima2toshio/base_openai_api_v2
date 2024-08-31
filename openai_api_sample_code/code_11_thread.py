# [beta] Thread
from openai import OpenAI

client = OpenAI()


def create_thread():
    empty_thread = client.beta.threads.create()
    return empty_thread


def retrieve_thread():
    my_thread = client.beta.threads.retrieve("thread_abc123")
    return my_thread


def update_thread():
    my_updated_thread = client.beta.threads.update(
        "thread_abc123",
        metadata={
            "modified": "true",
            "user": "abc123"
        }
    )
    return my_updated_thread


def delete_thread():
    response = client.beta.threads.delete("thread_abc123")
    return response


"""
response = 
  "id": "thread_abc123",
  "object": "thread",
  "created_at": 1699012949,
  "metadata": {},
  "tool_resources": {}
}
"""
