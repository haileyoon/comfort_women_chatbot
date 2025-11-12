import requests
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from src.tools import get_response

def chatbot_response(user_input: str, chat_history=None):
    if chat_history is None:
        chat_history = []
    load_dotenv()
    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
    functions = [{
        "type": "function",
        "name": "get_response",
        "description": "Get the information about comfort women that is most appropriate for the user's question",
        "parameters":{
            "type":"object",
            "properties": {
                "question" : {"type":"string","description":"question about comfort women"}
            },
            "required": ["question"],
        },
    }]
    # ensure system message is first so model knows function usage rules
    system_msg = {"role":"system","content":"You can call a function to return information about comfort women that is most appropriate for the user's question"}
    input_messages = [system_msg] + chat_history + [
        {"role":"user","content": user_input}
    ]
    try:
        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = input_messages,
            functions = functions,
            function_call = "auto",
        )
    except Exception as e:
        # surface API errors clearly
        return f"Error calling LLM: {e}"

    message = response.choices[0].message
    if message.function_call:
        func_name = message.function_call.name
        args = json.loads(message.function_call.arguments)
        if func_name=="get_response":
            pinecone_results = get_response(**args)
            # debug: print raw pinecone results (only serializable parts)
            serializable_matches = []
            for match in pinecone_results.get('matches', []):
                if isinstance(match, dict):
                    serializable_matches.append(match)
                elif hasattr(match, 'to_dict'):
                    serializable_matches.append(match.to_dict())
                else:
                    serializable_matches.append(str(match))
            print("[DEBUG] pinecone_results (matches):", json.dumps(serializable_matches, ensure_ascii=False, indent=2))
            top_chunks = []
            for match in pinecone_results.get('matches', []):
                # Some clients return 'metadata' or 'fields' - check both
                metadata = match.get('metadata', {}) or match.get('fields', {})
                chunk = metadata.get('chunk_text', '')
                if not chunk:
                    # try top-level field
                    chunk = match.get('chunk_text', '')
                if chunk:
                    top_chunks.append(chunk)
            # debug: print extracted chunks
            print("[DEBUG] top_chunks:", top_chunks)
            # prepare a plain text concatenation for the function result so the LLM can read it easily
            combined = "\n\n".join(top_chunks) if top_chunks else ""
            function_response = {"response": top_chunks, "response_text": combined}
        else:
            return "Unknown function call: " + func_name
        # build followup messages with system first, then history, then user, then the function result
        followup_messages = [
            {"role": "system", "content": "Reply to the user with the most helpful and relevant information. Use the information provided by the function call as your main source; you may supplement only if necessary."}
        ] + chat_history + [
            {"role": "user", "content": user_input},
            message,
            {
                "role": "function",
                "name": func_name,
                "content": function_response["response_text"]
            }
        ]
        try:
            followup = client.chat.completions.create(
                model = "gpt-4o-mini",
                messages = followup_messages,
            )
        except Exception as e:
            return f"Error calling LLM for followup: {e}"
        return followup.choices[0].message.content
    else:
        return message.content

if __name__ == "__main__":
    chat_history = []
    while True:
        user_input = input("User:")
        if(user_input.lower() in ["exit", "end","bye"]):
            print("Bot: Goodbye!")
            break
        chat_history.append({"role": "user", "content": user_input})
        response = chatbot_response(user_input, chat_history)
        chat_history.append({"role": "assistant", "content": response})
        print("Bot:", response)
