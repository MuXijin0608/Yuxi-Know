from dataclasses import dataclass, field
from typing import Annotated

from langchain.agents import create_agent

from src.agents.common import BaseAgent, BaseContext, gen_tool_info, load_chat_model
from src.agents.common.tools import get_buildin_tools
from src.agents.common.middlewares import RuntimeConfigMiddleware
from src.services.mcp_service import get_tools_from_all_servers


def _get_your_agent_tool_options() -> list[dict]:
    allowed_names = {"agentic_rag_search"}
    tools = [tool for tool in get_buildin_tools() if tool.name in allowed_names]
    return gen_tool_info(tools)

PROMPT = """你是一个面向意图感知检索的 Agent。
当用户问题涉及知识检索时，必须优先调用工具 `agentic_rag_search`，不要直接调用 `simkgc_pg_search` 或其他检索工具。
返回答案时，请先给出路由结果（bm25/simkgc/hybrid）和状态路径，再给出结论。
"""


@dataclass(kw_only=True)
class YourAgentContext(BaseContext):
    system_prompt: Annotated[str, {"__template_metadata__": {"kind": "prompt"}}] = field(
        default=PROMPT,
        metadata={"name": "系统提示词", "description": "Agentic RAG 的系统提示词"},
    )

    tools: Annotated[list[dict], {"__template_metadata__": {"kind": "tools"}}] = field(
        default_factory=lambda: ["agentic_rag_search"],
        metadata={
            "name": "工具",
            "options": _get_your_agent_tool_options,
            "description": "默认仅启用 agentic_rag_search，用于强制路由。",
        },
    )


class YourAgent(BaseAgent):
    name = "Custom Agent"
    description = "最小模板智能体：支持通过配置选择模型、工具、知识库和 MCP。"
    context_schema = YourAgentContext

    async def get_graph(self, **kwargs):
        context = self.context_schema.from_file(module_name=self.module_name)
        all_mcp_tools = await get_tools_from_all_servers()

        graph = create_agent(
            model=load_chat_model(context.model),
            system_prompt=context.system_prompt,
            middleware=[
                RuntimeConfigMiddleware(extra_tools=all_mcp_tools),
            ],
            checkpointer=await self._get_checkpointer(),
        )

        return graph
