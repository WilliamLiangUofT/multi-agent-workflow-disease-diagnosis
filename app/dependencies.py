import contextlib
from core.mcp_client import MCPPubMedClient
from core.tools_loader import get_structured_tools
from fastapi import FastAPI, Request

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mcp_client = MCPPubMedClient("http://127.0.0.1:8000/mcp") # Establish the MCP server connection through MCP client
    await app.state.mcp_client.__aenter__()
    app.state.tools = await get_structured_tools(client=app.state.mcp_client)
    yield
    await app.state.mcp_client.__aexit__(None, None, None)

async def get_tools(request: Request):
    return request.app.state.tools
