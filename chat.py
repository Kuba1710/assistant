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

load_dotenv()

recall_vector_store = Chroma(
    persist_directory="./memory_db",
    embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
)

def get_spotify_client():
    """Get Spotify client with proper authentication handling."""
    try:
        # Try to use existing token first
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

tools = [save_recall_memory, search_recall_memories, search_internet, spotify]

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

