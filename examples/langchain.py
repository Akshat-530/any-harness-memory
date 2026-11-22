"""Use Memory with LangChain / LangGraph via the MCP adapters.

    pip install langchain-mcp-adapters langgraph
"""
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main() -> None:
    client = MultiServerMCPClient({
        "memory": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:4000/mcp",
            # With token auth:
            # "headers": {"Authorization": "Bearer <token>"},
        }
    })
    tools = await client.get_tools()
    print(f"{len(tools)} memory tools available")

    # Example: remember then recall through the tools directly.
    store = next(t for t in tools if t.name == "memory_store")
    recall = next(t for t in tools if t.name == "memory_recall")
    await store.ainvoke({"content": "LangChain wrote this memory.",
                         "category": "fact", "importance": 5})
    result = await recall.ainvoke({"query": "LangChain memory", "top_k": 3})
    print(result)


if __name__ == "__main__":
    asyncio.run(main())