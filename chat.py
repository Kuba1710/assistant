#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

def load_api_key():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    return api_key

def initialize_client(api_key):
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

def get_ai_response(client, messages, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI response: {e}"

def main():
    
    api_key = load_api_key()
    client = initialize_client(api_key)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise and friendly in your responses."}
    ]
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n Goodbye!")
                break
            
            # Check for clear command
            if user_input.lower() == 'clear':
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Be concise and friendly in your responses."}
                ]
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            print("AI: ", end="", flush=True)
            ai_response = get_ai_response(client, messages)
            print(ai_response)
            
            messages.append({"role": "assistant", "content": ai_response})
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Unexpected error: {e}")

if __name__ == "__main__":
    main() 