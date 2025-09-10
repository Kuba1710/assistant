#!/usr/bin/env python3
import getpass
import os
import json
from typing import List, Literal, Optional
import uuid
from typing_extensions import TypedDict
from dotenv import load_dotenv
import spotipy
import tiktoken
from langchain_core.documents import Document
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from spotipy.oauth2 import SpotifyOAuth
from todoist_api_python.api import TodoistAPI
from datetime import datetime, timedelta
import re

load_dotenv()

recall_vector_store = Chroma(
    persist_directory="./memory_db",
    embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
)

def get_spotify_client():
    """Get Spotify client with proper authentication handling."""
    try:
        auth_manager = SpotifyOAuth(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
            redirect_uri="http://127.0.0.1:8000/callback",
            scope="user-read-playback-state,user-modify-playback-state,playlist-read-private",
            cache_path=".spotify_cache",
            open_browser=False
        )

        if os.path.exists(".spotify_cache"):
            return spotipy.Spotify(auth_manager=auth_manager)
        else:
            return None
    except Exception as e:
        return None

sp = get_spotify_client()

def get_todoist_client():
    """Get Todoist client with proper authentication handling."""
    try:
        api_key = os.getenv("TODOIST_API_KEY")
        if api_key:
            return TodoistAPI(api_key)
        else:
            return None
    except Exception as e:
        return None

todoist = get_todoist_client()


def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id

class KnowledgeTriple(TypedDict):
    subject: str
    predicate: str
    object_: str


@tool
def save_recall_memory(memories: List[KnowledgeTriple], config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    for memory in memories:
        serialized = " ".join(memory.values())
        document = Document(
            serialized,
            id=str(uuid.uuid4()),
            metadata={
                "user_id": user_id,
                **memory,
            },
        )
        recall_vector_store.add_documents([document])
    return memories


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)

    documents = recall_vector_store.similarity_search(
        query, k=3, filter={"user_id": user_id}
    )
    return [document.page_content for document in documents]

@tool
def spotify(query: str) -> str:
    """Search and play a song on Spotify."""
    try:
        results = sp.search(q=query, type="track", limit=1)
        
        if not results["tracks"]["items"]:
            return f"No tracks found for '{query}'"
        
        track = results["tracks"]["items"][0]
        track_uri = track["uri"]
        track_name = track["name"]
        artist_name = track["artists"][0]["name"]
        
        sp.start_playback(uris=[track_uri])
        return f"Now playing: {track_name} by {artist_name}"
        
    except Exception as e:
        return f"Error playing song: {str(e)}. Make sure Spotify is open and you have an active device."

tavily_search = TavilySearch(max_results=3, api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_internet(query: str) -> str:
    """Search the internet for current information about any topic."""
    try:
        results = tavily_search.run(query)
        if results:
            return results
        else:
            return f"No results found for '{query}'"
    except Exception as e:
        return f"Error searching the internet: {str(e)}. Please check your TAVILY_API_KEY in the .env file."

@tool
def list_todoist_projects() -> str:
    """List all available Todoist projects."""
    if not todoist:
        return "Todoist is not configured. Please add your TODOIST_API_KEY to the .env file."
    
    try:
        projects_response = todoist.get_projects()
        projects_list = list(projects_response)
        projects = projects_list[0] if projects_list else []
        if not projects:
            return "No projects found in your Todoist account."
        
        project_list = "\n".join([f"- {project.name} (ID: {project.id})" for project in projects])
        return f"Available Todoist projects:\n{project_list}"
        
    except Exception as e:
        return f"Error fetching Todoist projects: {str(e)}"

@tool
def add_todoist_task(task_content: str, project_name: str = "Inbox", due_date: str = None) -> str:
    """Add a task to Todoist with optional project and due date.
    
    Args:
        task_content: The task description
        project_name: Name of the project (default: "Inbox")
        due_date: Due date in natural language (e.g., "tomorrow", "today", "next week", "2024-12-25")
    """
    if not todoist:
        return "Todoist is not configured. Please add your TODOIST_API_KEY to the .env file."
    
    try:

        projects_response = todoist.get_projects()
        projects_list = list(projects_response)
        projects = projects_list[0] if projects_list else []
        project_id = None
        
        for project in projects:
            if project.name.lower() == project_name.lower():
                project_id = project.id
                break
            elif project_name.lower() in project.name.lower():
                project_id = project.id
                break
        
        if project_id is None and project_name.lower() != "inbox":
            available = ', '.join([p.name for p in projects])
            return f"Project '{project_name}' not found. Available projects: {available}"
        
        due_string = due_date.strip() if due_date else None
        
        task_data = {
            "content": task_content,
        }
        
        if project_id is not None:
            task_data["project_id"] = project_id
        
        if due_string:
            task_data["due_string"] = due_string
        
        task = todoist.add_task(**task_data)
        
        project_display = project_name if project_id else "Inbox"
        due_display = f" (due: {due_string})" if due_string else ""
        
        return f"✅ Task added successfully to '{project_display}': {task_content}{due_display}"
        
    except Exception as e:
        return f"Error adding task to Todoist: {str(e)}"

tools = [save_recall_memory, search_recall_memories, search_internet, spotify, add_todoist_task, list_todoist_projects]

class State(MessagesState):
    recall_memories: List[str]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (save_core_memory, save_recall_memory)"
            " to build a comprehensive understanding of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored"
            " memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "4. Update your mental model of the user with each new piece of"
            " information.\n"
            "5. Cross-reference new information with existing memories for"
            " consistency.\n"
            "6. Prioritize storing emotional context and personal values"
            " alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the"
            " user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and"
            " analogies.\n"
            "10. Recall past challenges or successes to inform current"
            " problem-solving.\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            "## Available Tools\n"
            "You have access to several tools:\n"
            "- Memory tools: save and search personal memories\n"
            "- Spotify: search and play music\n"
            "- Internet search: get current information\n"
            "- Todoist: add tasks to projects with due dates\n\n"
            "For Todoist tasks, you can understand natural language requests like:\n"
            "- 'Tomorrow I need to clean my home' -> add task 'clean my home' to 'home' project for tomorrow\n"
            "- 'Add buy groceries to my shopping list for Friday' -> add to shopping project for Friday\n"
            "- 'Remind me to call mom next week' -> add task with due date next week\n\n"
            "## Instructions\n"
            "Engage with the user naturally, as a trusted colleague or friend."
            " There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses. Be attentive to subtle cues and underlying"
            " emotions. Adapt your communication style to match the user's"
            " preferences and current emotional state. Use tools to persist"
            " information you want to retain in the next conversation. If you"
            " do call tools, all text preceding the tool call is an internal"
            " message. Respond AFTER calling the tool, once you have"
            " confirmation that the tool completed successfully.\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)

model = ChatOpenAI(
    model_name="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)
model_with_tools = model.bind_tools(tools)

tokenizer = tiktoken.encoding_for_model("gpt-4o")


def agent(state: State) -> State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    bound = prompt | model_with_tools
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = bound.invoke(
        {
            "messages": state["messages"],
            "recall_memories": recall_str,
        }
    )
    return {
        "messages": [prediction],
    }


def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        State: The updated state with loaded memories.
    """
    convo_str = get_buffer_string(state["messages"])
    try:
        tokens = tokenizer.encode(convo_str)
        if len(tokens) > 2048:
            truncated_tokens = tokens[:2048]
            convo_str = tokenizer.decode(truncated_tokens)
        convo_str = convo_str.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception as e:
        convo_str = convo_str[:4000]
        convo_str = convo_str.encode('utf-8', errors='ignore').decode('utf-8')
    
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {
        "recall_memories": recall_memories,
    }


def route_tools(state: State):
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"

    return END

builder = StateGraph(State)
builder.add_node(load_memories)
builder.add_node(agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools, ["tools", END])
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"user_id": "1", "thread_id": "1"}}

def interactive_chat():
    """Interactive chat loop with memory capabilities."""
    print("AI Assistant with Memory")
    
    while True:
        try:
            user_input = input("\n😊 You: ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
            
        print(f"\n AI: ", end="", flush=True)
        
        
        for chunk in graph.stream({"messages": [("user", user_input)]}, config=config):
            for node, updates in chunk.items():
                if "messages" in updates:
                    message = updates["messages"][-1]
                    
                    if node == "agent":
                        if hasattr(message, 'content') and message.content:
                            print(message.content)
                    
                elif node == "tools":
                    continue
                else:
                    if node == "load_memories" and "recall_memories" in updates:
                        memories = updates["recall_memories"]

if __name__ == "__main__":
    interactive_chat()

