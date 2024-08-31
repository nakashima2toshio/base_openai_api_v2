import gradio as gr
from openai import OpenAI
import json

model_gpt_4o_mini = 'gpt-4o-mini'
client = OpenAI()

def get_current_weather(location, unit="fahrenheit"):
    # """指定された場所の現在の天気を取得する"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation(user_input):
    messages = [{"role": "user", "content": user_input}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "指定された場所の現在の天気を取得する",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "都市名と州名、例: San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model=model_gpt_4o_mini,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = {
            "get_current_weather": get_current_weather,
        }
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        second_response = client.chat.completions.create(
            model=model_gpt_4o_mini,
            messages=messages,
        )
        return second_response.choices[0].message.content
    else:
        return response_message.content

def chatbot(input_text):
    response = run_conversation(input_text)
    return response

def main():
    iface = gr.Interface(
        fn=chatbot,
        inputs=gr.Textbox(lines=2, placeholder="天気について質問してください..."),
        outputs="text",
        title="天気チャットボット",
        description="都市名を入力して天気情報を尋ねてください。例: 'What's the weather like in Tokyo?'",
    )

    iface.launch()

if __name__ == "__main__":
    main()
