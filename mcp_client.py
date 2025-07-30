#import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from contextlib import AsyncExitStack
from typing import Any, Optional
from mcp.types import Tool as MCPTool

class MCPPubMedClient:
    def __init__(self, url: str):
        self.session: Optional[ClientSession] = None
        self.read_stream: Optional[Any] = None
        self.write_stream: Optional[Any] = None
        self.exit_stack = AsyncExitStack()
        self.url = url

    async def __aenter__(self):
        # Connect to the server
        self.read_stream, self.write_stream, _ = await self.exit_stack.enter_async_context(
            streamablehttp_client(self.url)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read_stream, self.write_stream)
        )

        # Initialize server connection
        await self.session.initialize()

        return self # with ContextManager() as obj, where obj will be this self, the MCPPubMedClient()
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.exit_stack.aclose()
        
    async def print_mcp_tools(self):
        all_tools = await self.session.list_tools()

        print("Available Tools: ")
        for tool in all_tools.tools:
            print(f"- {tool.name}: {tool.description}")

    async def list_mcp_tools(self) -> list[MCPTool]:
        listed_tools = await self.session.list_tools() 
        all_tools = []

        for tool in listed_tools.tools:
            all_tools.append(tool)

        return all_tools # list[MCPTool]


# async def main(): # DONT DO MAIN IN THIS FILE. IT WILL BE IN THE ACTUALLY FINAL LANGGRAPH APP FILE. APP FILE IMPORTS MCPPubMedClient
#     client = MCPPubMedClient("http://127.0.0.1:8000/mcp")
#     try:
#         await client.connect_to_mcp_server()
#         await client.print_mcp_tools()

#         print("calling tool")
#         result = await client.session.call_tool("search_pubmed_key_words", arguments={'key_words': 'CRISPR', 'num_results': 2}) # SHORT KEYWORD SEARCH
#         #print(result.content[0].text) # just .content is a list of TextContent (each an article)
#         # HANDLE EXCEPTION FOR THIS IN APP. THIS CAN THROW EXCEPTION IF SOME PARSING GLITCHES LIKE BEFORE, USUALLY WONT HAPPEN PROBABLY
#         #print(result) # meta=None content=[] structuredContent={'result': []} isError=False
#         # CONTENT IS A LIST OF TEXTCONTENT, EACH REPRESENT AN ARTICLE. HOWEVER, SOMETIMES IT CAN BE EMPTY [] IF NOTHING WAS FOUND
#     finally:
#         await client.exit_stack.aclose()
    
# if __name__ == "__main__":
#     asyncio.run(main())

# async def main():
#     url = "http://127.0.0.1:8000/mcp"

#     async with streamablehttp_client(url) as (read_stream, write_stream, get_session_id):
#         async with ClientSession(read_stream, write_stream) as session:
#             await session.initialize()

#             all_tools = await session.list_tools()
#             print("Available Tools: ")
            
#             for tool in all_tools.tools:
#                 print(f"- {tool.name}: {tool.description}")

