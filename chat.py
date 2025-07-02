#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

MEMORY_DIR = "memory"
CONVERSATION_FILE = os.path.join(MEMORY_DIR, "conversation_history.json")
USER_PROFILE_FILE = os.path.join(MEMORY_DIR, "user_profile.json")

def ensure_memory_directory():
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR)

def save_conversation_history(messages):
    ensure_memory_directory()
    conversation_data = []
    
    for message in messages:
        if isinstance(message, HumanMessage):
            conversation_data.append({
                "type": "human",
                "content": message.content,
                "timestamp": datetime.now().isoformat()
            })
        elif isinstance(message, AIMessage):
            conversation_data.append({
                "type": "ai", 
                "content": message.content,
                "timestamp": datetime.now().isoformat()
            })
    
    try:
        with open(CONVERSATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save conversation history: {e}")

def load_conversation_history():
    if not os.path.exists(CONVERSATION_FILE):
        return []
    
    try:
        with open(CONVERSATION_FILE, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
        
        messages = []
        for item in conversation_data:
            if item["type"] == "human":
                messages.append(HumanMessage(content=item["content"]))
            elif item["type"] == "ai":
                messages.append(AIMessage(content=item["content"]))
        
        return messages
    except Exception as e:
        print(f"Warning: Could not load conversation history: {e}")
        return []

def save_user_profile(profile_data):
    """Save user profile information to JSON file"""
    ensure_memory_directory()
    
    try:
        existing_profile = {}
        if os.path.exists(USER_PROFILE_FILE):
            with open(USER_PROFILE_FILE, 'r', encoding='utf-8') as f:
                existing_profile = json.load(f)
        
        # Merge new data with existing
        existing_profile.update(profile_data)
        existing_profile["last_updated"] = datetime.now().isoformat()
        
        with open(USER_PROFILE_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_profile, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save user profile: {e}")

def load_user_profile():
    """Load user profile information from JSON file"""
    if not os.path.exists(USER_PROFILE_FILE):
        return {}
    
    try:
        with open(USER_PROFILE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load user profile: {e}")
        return {}

def extract_user_info(message_content):
    """Extract user information from message content"""
    content_lower = message_content.lower()
    profile_updates = {}
    
    # Extract name
    if "my name is" in content_lower or "i'm" in content_lower or "i am" in content_lower:
        words = message_content.split()
        for i, word in enumerate(words):
            if word.lower() in ["is", "i'm", "am"] and i + 1 < len(words):
                potential_name = words[i + 1].strip(".,!?")
                if potential_name.isalpha() and len(potential_name) > 1:
                    profile_updates["name"] = potential_name
                    break
    
    # Extract age
    if "i am" in content_lower and "years old" in content_lower:
        import re
        age_match = re.search(r'i am (\d+) years old', content_lower)
        if age_match:
            profile_updates["age"] = int(age_match.group(1))
    
    # Extract location
    if "i live in" in content_lower or "from" in content_lower:
        if "i live in" in content_lower:
            location = content_lower.split("i live in")[1].split()[0].strip(".,!?")
            if location:
                profile_updates["location"] = location.title()
        elif "from" in content_lower:
            words = message_content.split()
            from_index = next((i for i, word in enumerate(words) if word.lower() == "from"), -1)
            if from_index != -1 and from_index + 1 < len(words):
                location = words[from_index + 1].strip(".,!?")
                if location.isalpha():
                    profile_updates["location"] = location.title()
    
    return profile_updates

class PersistentChatMessageHistory(InMemoryChatMessageHistory):
    """Custom chat history that persists to file"""
    
    def __init__(self):
        super().__init__()
        # Load existing messages
        self.messages = load_conversation_history()
    
    def add_message(self, message):
        super().add_message(message)
        # Save after each message
        save_conversation_history(self.messages)
        
        # Extract and save user information
        if isinstance(message, HumanMessage):
            user_info = extract_user_info(message.content)
            if user_info:
                save_user_profile(user_info)
    
    def clear(self):
        super().clear()
        # Clear persistent storage
        if os.path.exists(CONVERSATION_FILE):
            os.remove(CONVERSATION_FILE)

def load_api_key():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        sys.exit(1)
    
    return api_key

def initialize_chat_chain(api_key):
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Load user profile for context
        user_profile = load_user_profile()
        profile_context = ""
        if user_profile:
            # Format profile information without JSON to avoid template variable conflicts
            profile_info = []
            for key, value in user_profile.items():
                if key != "last_updated":
                    profile_info.append(f"{key}: {value}")
            if profile_info:
                profile_context = f"\n\nUser Profile Information:\n" + "\n".join(f"- {info}" for info in profile_info)
        
        # Create a prompt template with message history placeholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant with persistent memory. You can remember our conversations across sessions and user information.{profile_context}\n\nBe concise and friendly in your responses. Reference previous conversations and user information when relevant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create the chain
        chain = prompt | llm
        
        # Store for message history
        store = {}
        
        def get_session_history(session_id: str) -> PersistentChatMessageHistory:
            if session_id not in store:
                store[session_id] = PersistentChatMessageHistory()
            return store[session_id]
        
        # Create conversation with message history
        conversation = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        return conversation, store
    except Exception as e:
        print(f"Error initializing LangChain conversation: {e}")
        sys.exit(1)

def main():
    print("🧠 AI Chat with Persistent Memory")
    print("📁 Your conversations and information are saved to 'memory/' folder")
    print("Commands: 'quit', 'exit', 'bye' to exit | 'clear' to clear memory | 'memory' to see history | 'profile' to see your info")
    print("-" * 80)
    
    api_key = load_api_key()
    conversation, store = initialize_chat_chain(api_key)
    session_id = "user_session"
    
    # Show loaded profile if exists
    user_profile = load_user_profile()
    if user_profile:
        print(f"\n✅ Loaded your profile: {json.dumps(user_profile, indent=2)}")
    
    # Show conversation count
    conversation_history = load_conversation_history()
    if conversation_history:
        print(f"📚 Loaded {len(conversation_history)} previous messages")
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye! I'll remember our conversation for next time.")
                break
            
            # Check for clear command
            if user_input.lower() == 'clear':
                if session_id in store:
                    store[session_id].clear()
                if os.path.exists(USER_PROFILE_FILE):
                    os.remove(USER_PROFILE_FILE)
                print("\n🧹 Memory cleared! All persistent data removed.")
                continue
            
            # Check for profile command
            if user_input.lower() == 'profile':
                user_profile = load_user_profile()
                if user_profile:
                    print(f"\n👤 Your Profile:")
                    for key, value in user_profile.items():
                        if key != "last_updated":
                            print(f"   {key.title()}: {value}")
                    print(f"   Last Updated: {user_profile.get('last_updated', 'Unknown')}")
                else:
                    print("\n👤 No profile information saved yet.")
                continue
            
            if user_input.lower() == 'memory':
                print("\n📚 Conversation History:")
                if session_id in store and store[session_id].messages:
                    for message in store[session_id].messages:
                        if isinstance(message, HumanMessage):
                            print(f"👤 You: {message.content}")
                        elif isinstance(message, AIMessage):
                            print(f"🤖 AI: {message.content}")
                else:
                    print("No conversation history yet.")
                continue
            
            print("\n🤖 AI: ", end="", flush=True)
            ai_response = conversation.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(ai_response.content)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! I'll remember our conversation for next time.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("The conversation continues, but this message wasn't processed.")

if __name__ == "__main__":
    main() 