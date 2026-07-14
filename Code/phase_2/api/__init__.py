"""Serving doors (build-order phase 7). Deliberately NOT imported by the
phase_2 package root - fastapi/mcp are serving-only dependencies:

    rest        FastAPI: POST rows -> scores; same pipeline engine
                run:  uvicorn phase_2.api.rest:app  (from Code/)
    mcp_server  MCP tools over the same gated engine - any MCP client
                (Claude Desktop, agents) can operate ChurnLens
                run:  python -m phase_2.api.mcp_server  (from Code/)
"""
