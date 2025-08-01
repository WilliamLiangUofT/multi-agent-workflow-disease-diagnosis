from core.mcp_client import MCPPubMedClient
from typing import Any
from mcp.types import Tool as MCPTool
from langchain_core.tools.structured import StructuredTool

async def get_structured_tools(client: MCPPubMedClient) -> list[StructuredTool]:
    all_tools = await client.list_mcp_tools() # list[MCPTool]
    search_keyword_tool = [tool for tool in all_tools if tool.name == "search_pubmed_key_words"][0] # MCPTool

    async def convert_mcp_tool_to_langchain_tool(tool: MCPTool):
        async def tool_call(**arguments: dict[str, Any]):
            tool_call_result = await client.session.call_tool(tool.name, arguments)
            str_tool_call_result = [article.text for article in tool_call_result.content] # This would make this specific to search_keyword_tool tool
            return str_tool_call_result
            
        return StructuredTool(
            name=tool.name,
            description=tool.description or "",
            args_schema=tool.inputSchema, # What arguments tool takes
            coroutine=tool_call # How the ToolNode/agent can actually call the function
        )
        
    pubmed_search_tool = [await convert_mcp_tool_to_langchain_tool(search_keyword_tool)] # loop convert_mcp_tool_to_langchain_tool across MCPTools to get many StructuredTools

    # result = await pubmed_search_tool[0].ainvoke({'key_words': 'CRISPR', 'num_results': 2})
    # print(result)

    return pubmed_search_tool
    