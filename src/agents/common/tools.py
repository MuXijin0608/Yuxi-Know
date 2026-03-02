import asyncio
import math
import os
import re
import sys
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import requests
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src import config, graph_base, knowledge_base
from src.services.mcp_service import get_enabled_mcp_tools
from src.storage.minio import aupload_file_to_minio
from src.utils import logger

# Lazy initialization for TavilySearch (only when TAVILY_API_KEY is available)
_tavily_search_instance = None


def get_tavily_search():
    """Get TavilySearch instance lazily, only when API key is available."""
    global _tavily_search_instance
    if _tavily_search_instance is None and config.enable_web_search:
        from langchain_tavily import TavilySearch

        _tavily_search_instance = TavilySearch()
        _tavily_search_instance.metadata = {"name": "Tavily 网页搜索"}
    return _tavily_search_instance


@tool(name_or_callable="calculator", description="可以对给定的2个数字选择进行 add, subtract, multiply, divide 运算")
def calculator(a: float, b: float, operation: str) -> float:
    try:
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ZeroDivisionError("除数不能为零")
            return a / b
        else:
            raise ValueError(f"不支持的运算类型: {operation}，仅支持 add, subtract, multiply, divide")
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        raise


@tool(name_or_callable="my_custom_tool", description="示例自定义工具：返回格式化的问候语")
def my_custom_tool(name: str, topic: str = "智能体") -> str:
    return f"你好，{name}！这是你的自定义工具，当前主题是：{topic}。"


@tool
async def text_to_img_demo(text: str) -> str:
    """【测试用】使用模型生成图片， 会返回图片的URL"""

    url = "https://api.siliconflow.cn/v1/images/generations"

    payload = {
        "model": "Qwen/Qwen-Image",
        "prompt": text,
    }
    headers = {"Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}", "Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()
    except Exception as e:
        logger.error(f"Failed to generate image with: {e}")
        raise ValueError(f"Image generation failed: {e}")

    try:
        image_url = response_json["images"][0]["url"]
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Failed to parse image URL from response: {e}, {response_json=}")
        raise ValueError(f"Image URL extraction failed: {e}")

    # 2. Upload to MinIO (Simplified)
    response = requests.get(image_url)
    file_data = response.content

    file_name = f"{uuid.uuid4()}.jpg"
    image_url = await aupload_file_to_minio(
        bucket_name="generated-images", file_name=file_name, data=file_data, file_extension="jpg"
    )
    logger.info(f"Image uploaded. URL: {image_url}")
    return image_url


@tool(name_or_callable="human_in_the_loop_debug", description="请求人工审批工具，用于在执行重要操作前获得人类确认。")
def get_approved_user_goal(
    operation_description: str,
) -> dict:
    """
    请求人工审批，在执行重要操作前获得人类确认。

    Args:
        operation_description: 需要审批的操作描述，例如 "调用知识库工具"
    Returns:
        dict: 包含审批结果的字典，格式为 {"approved": bool, "message": str}
    """
    # 构建详细的中断信息
    interrupt_info = {
        "question": "是否批准以下操作？",
        "operation": operation_description,
    }

    # 触发人工审批
    is_approved = interrupt(interrupt_info)

    # 返回审批结果
    if is_approved:
        result = {
            "approved": True,
            "message": f"✅ 操作已批准：{operation_description}",
        }
        print(f"✅ 人工审批通过: {operation_description}")
    else:
        result = {
            "approved": False,
            "message": f"❌ 操作被拒绝：{operation_description}",
        }
        print(f"❌ 人工审批被拒绝: {operation_description}")

    return result


KG_QUERY_DESCRIPTION = """
使用这个工具可以查询知识图谱中包含的三元组信息。
关键词（query），使用可能帮助回答这个问题的关键词进行查询，不要直接使用用户的原始输入去查询。
"""


@tool(name_or_callable="查询知识图谱", description=KG_QUERY_DESCRIPTION)
def query_knowledge_graph(query: Annotated[str, "The keyword to query knowledge graph."]) -> Any:
    """使用这个工具可以查询知识图谱中包含的三元组信息。关键词（query），使用可能帮助回答这个问题的关键词进行查询，不要直接使用用户的原始输入去查询。"""
    try:
        logger.debug(f"Querying knowledge graph with: {query}")
        result = graph_base.query_node(query, hops=2, return_format="triples")
        logger.debug(
            f"Knowledge graph query returned "
            f"{len(result.get('triples', [])) if isinstance(result, dict) else 'N/A'} triples"
        )
        return result
    except Exception as e:
        logger.error(f"Knowledge graph query error: {e}, {traceback.format_exc()}")
        return f"知识图谱查询失败: {str(e)}"


class KnowledgeRetrieverModel(BaseModel):
    query_text: str = Field(
        description=(
            "查询的关键词，查询的时候，应该尽量以可能帮助回答这个问题的关键词进行查询，不要直接使用用户的原始输入去查询。"
        )
    )
    operation: str = Field(
        default="search",
        description=(
            "操作类型：'search' 表示检索知识库内容，'get_mindmap' 表示获取知识库的思维导图结构。"
            "当用户询问知识库的整体结构、文件分类、知识架构时，使用 'get_mindmap'。"
            "当用户需要查询具体内容时，使用 'search'。"
        ),
    )


class CommonKnowledgeRetriever(KnowledgeRetrieverModel):
    """Common knowledge retriever model."""

    file_name: str = Field(description="限定文件名称，当操作类型为 'search' 时，可以指定文件名称，支持模糊匹配")



class HybridRagSearchInput(BaseModel):
    query_text: str = Field(description="检索关键词")
    db_names: list[str] | None = Field(default=None, description="限定知识库名称列表，为空时检索全部知识库")
    include_graph: bool = Field(default=False, description="是否附带知识图谱检索结果")


@tool(
    name_or_callable="hybrid_rag_search",
    description="混合检索工具：复用系统内置RAG检索，并可选附带知识图谱结果。",
    args_schema=HybridRagSearchInput,
)
async def hybrid_rag_search(
    query_text: str,
    db_names: list[str] | None = None,
    include_graph: bool = False,
) -> dict[str, Any]:
    """最小混合检索实现：优先复用系统现有 RAG。"""
    retrievers = knowledge_base.get_retrievers()

    if db_names:
        selected_db_ids = [db_id for db_id, info in retrievers.items() if info.get("name") in db_names]
    else:
        selected_db_ids = list(retrievers.keys())

    kb_results: list[dict[str, Any]] = []
    for db_id in selected_db_ids:
        info = retrievers.get(db_id)
        if not info:
            continue

        retriever = info.get("retriever")
        if not callable(retriever):
            continue

        try:
            if asyncio.iscoroutinefunction(retriever):
                result = await retriever(query_text)
            else:
                result = retriever(query_text)
            kb_results.append({"db_id": db_id, "db_name": info.get("name"), "result": result})
        except Exception as e:
            logger.warning(f"hybrid_rag_search retriever failed ({db_id}): {e}")

    graph_results = None
    if include_graph:
        try:
            graph_results = graph_base.query_node(query_text, hops=2, return_format="triples")
        except Exception as e:
            logger.warning(f"hybrid_rag_search graph query failed: {e}")
            graph_results = {"error": str(e)}

    return {
        "query": query_text,
        "knowledge_results": kb_results,
        "graph_results": graph_results,
    }


def _bm25_tokenize(text: str) -> list[str]:
    lower_text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lower_text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lower_text)
    tokens.extend(cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1))
    return tokens


async def _collect_rag_chunks_for_bm25() -> list[dict[str, Any]]:
    chunk_rows: list[dict[str, Any]] = []

    for kb_instance in knowledge_base.kb_instances.values():
        files_meta = getattr(kb_instance, "files_meta", {}) or {}

        for file_id, file_meta in files_meta.items():
            if file_meta.get("is_folder"):
                continue

            status = str(file_meta.get("status") or "").lower()
            if status not in {"indexed", "done"}:
                continue

            db_id = file_meta.get("database_id")
            if not db_id:
                continue

            try:
                content_info = await knowledge_base.get_file_content(db_id, file_id)
            except Exception as e:
                logger.warning(f"BM25 get_file_content failed for file_id={file_id}: {e}")
                continue

            lines = content_info.get("lines") or []
            if not isinstance(lines, list):
                continue

            for idx, line in enumerate(lines):
                if not isinstance(line, dict):
                    continue

                content = str(line.get("content") or line.get("text") or "").strip()
                if not content:
                    continue

                chunk_id = str(line.get("id") or line.get("chunk_id") or f"{file_id}:{idx}")
                chunk_rows.append(
                    {
                        "db_id": db_id,
                        "db_name": kb_instance.databases_meta.get(db_id, {}).get("name", ""),
                        "file_id": file_id,
                        "filename": file_meta.get("filename", ""),
                        "chunk_id": chunk_id,
                        "chunk_order_index": line.get("chunk_order_index", idx),
                        "content": content,
                    }
                )

    return chunk_rows


async def _run_bm25_search_stub(query_text: str) -> dict[str, Any]:
    chunks = await _collect_rag_chunks_for_bm25()
    if not chunks:
        return {
            "ok": False,
            "engine": "bm25",
            "message": "未找到可检索的已切分数据（请确认知识库已完成索引）。",
            "query": query_text,
            "results": [],
        }

    query_tokens = _bm25_tokenize(query_text)
    if not query_tokens:
        return {
            "ok": False,
            "engine": "bm25",
            "message": "查询词为空或无法分词。",
            "query": query_text,
            "results": [],
        }

    docs_tokens: list[list[str]] = []
    docs_tf: list[Counter[str]] = []
    doc_lens: list[int] = []
    df: Counter[str] = Counter()

    for row in chunks:
        tokens = _bm25_tokenize(row["content"])
        docs_tokens.append(tokens)
        tf = Counter(tokens)
        docs_tf.append(tf)
        doc_lens.append(len(tokens))
        for token in set(tokens):
            df[token] += 1

    n_docs = len(chunks)
    avgdl = (sum(doc_lens) / n_docs) if n_docs else 1.0
    k1 = 1.5
    b = 0.75
    query_tf = Counter(query_tokens)

    scored: list[tuple[float, dict[str, Any]]] = []
    for idx, row in enumerate(chunks):
        tf = docs_tf[idx]
        dl = doc_lens[idx] or 1
        score = 0.0
        for token, qf in query_tf.items():
            freq = tf.get(token, 0)
            if freq == 0:
                continue
            token_df = df.get(token, 0)
            idf = math.log(1 + (n_docs - token_df + 0.5) / (token_df + 0.5))
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            score += qf * idf * ((freq * (k1 + 1)) / denom)

        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    topk = 10
    results = []
    for score, row in scored[:topk]:
        results.append(
            {
                "score": round(score, 6),
                "db_id": row["db_id"],
                "db_name": row["db_name"],
                "file_id": row["file_id"],
                "filename": row["filename"],
                "chunk_id": row["chunk_id"],
                "chunk_order_index": row["chunk_order_index"],
                "content": row["content"],
            }
        )

    return {
        "ok": True,
        "engine": "bm25",
        "query": query_text,
        "total_chunks": n_docs,
        "matched_chunks": len(scored),
        "results": results,
    }


class SimKGCPGSearchInput(BaseModel):
    query_text: str = Field(description="检索查询文本")
    chroma_dir: str = Field(default="chroma_db_full", description="Chroma 向量库目录")
    l1_collection: str = Field(default="simkgc_power_l1_full", description="L1 集合名")
    l0_collection: str = Field(default="simkgc_power_l0_full", description="L0 集合名")
    text_encoder: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="文本编码器模型")


@tool(
    name_or_callable="simkgc_pg_search",
    description="调用本地 SimKGC-PG 分层检索脚本，返回检索输出。",
    args_schema=SimKGCPGSearchInput,
)
async def simkgc_pg_search(
    query_text: str,
    chroma_dir: str = "chroma_db_full",
    l1_collection: str = "simkgc_power_l1_full",
    l0_collection: str = "simkgc_power_l0_full",
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    return await _run_simkgc_pg_search(
        query_text=query_text,
        chroma_dir=chroma_dir,
        l1_collection=l1_collection,
        l0_collection=l0_collection,
        text_encoder=text_encoder,
    )


async def _run_simkgc_pg_search(
    query_text: str,
    chroma_dir: str = "chroma_db_full",
    l1_collection: str = "simkgc_power_l1_full",
    l0_collection: str = "simkgc_power_l0_full",
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    script_dir = project_root / "scripts" / "simkgc_pg_runtime_nocache"
    script_path = script_dir / "query_hierarchical_retrieval.py"

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script_path),
        "--query",
        query_text,
        "--chroma-dir",
        chroma_dir,
        "--l1-collection",
        l1_collection,
        "--l0-collection",
        l0_collection,
        "--text-encoder",
        text_encoder,
        cwd=str(script_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    return {
        "ok": process.returncode == 0,
        "returncode": process.returncode,
        "stdout": stdout.decode("utf-8", errors="ignore"),
        "stderr": stderr.decode("utf-8", errors="ignore"),
    }


async def _run_system_rag_search(
    query_text: str,
    db_names: list[str] | None = None,
) -> dict[str, Any]:
    retrievers = knowledge_base.get_retrievers()
    if not retrievers:
        return {
            "ok": False,
            "engine": "system_rag",
            "query": query_text,
            "message": "系统尚未注册任何知识库检索器。",
            "results": [],
            "errors": [],
        }

    if db_names:
        target_names = {name.strip() for name in db_names if name and name.strip()}
        selected_items = [
            (db_id, info)
            for db_id, info in retrievers.items()
            if info.get("name") in target_names
        ]
    else:
        selected_items = list(retrievers.items())

    if not selected_items:
        return {
            "ok": False,
            "engine": "system_rag",
            "query": query_text,
            "message": "未匹配到指定名称的知识库。",
            "results": [],
            "errors": [],
        }

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for db_id, info in selected_items:
        retriever = info.get("retriever")
        if not callable(retriever):
            errors.append(f"知识库 {info.get('name', db_id)} 未提供 retriever 函数")
            continue

        try:
            if asyncio.iscoroutinefunction(retriever):
                payload = await retriever(query_text)
            else:
                payload = retriever(query_text)

            results.append(
                {
                    "db_id": db_id,
                    "db_name": info.get("name", ""),
                    "metadata": info.get("metadata", {}),
                    "result": payload,
                }
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            error_msg = f"检索 {info.get('name', db_id)} 失败: {exc}"
            logger.warning(error_msg)
            errors.append(error_msg)

    return {
        "ok": bool(results),
        "engine": "system_rag",
        "query": query_text,
        "results": results,
        "errors": errors,
        "selected_databases": [info.get("name", db_id) for db_id, info in selected_items],
    }


class AgenticRagSearchInput(BaseModel):
    query_text: str = Field(description="用户查询")
    route_hint: str | None = Field(
        default=None,
        description="可选路由提示：bm25 / simkgc / system / hybrid / all，或以 bm25+system 形式组合，不传则自动判定。",
    )
    simkgc_chroma_dir: str = Field(default="chroma_db_full", description="SimKGC Chroma 目录")
    simkgc_l1_collection: str = Field(default="simkgc_power_l1_full", description="SimKGC L1 集合")
    simkgc_l0_collection: str = Field(default="simkgc_power_l0_full", description="SimKGC L0 集合")
    simkgc_text_encoder: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SimKGC 文本编码器",
    )
    system_db_names: list[str] | None = Field(
        default=None,
        description="限定系统 RAG 检索的知识库名称列表，默认检索全部知识库",
    )


def _decide_agentic_rag_route(query_text: str, route_hint: str | None = None) -> str:
    valid_single = {"bm25", "simkgc", "system"}

    def _canonicalize(raw: str | None) -> str | None:
        if not raw:
            return None
        normalized = raw.strip().lower()
        if not normalized:
            return None
        if normalized in {"hybrid", "all"} | valid_single:
            return normalized

        tokens = [token.strip() for token in normalized.replace(",", "+").split("+") if token.strip()]
        if not tokens:
            return None

        canonical: list[str] = []
        for token in tokens:
            if token in valid_single and token not in canonical:
                canonical.append(token)

        if not canonical:
            return None

        if set(canonical) == {"bm25", "simkgc", "system"}:
            return "all"
        if set(canonical) == {"bm25", "simkgc"}:
            return "hybrid"
        return "+".join(canonical)

    canonical_hint = _canonicalize(route_hint)
    if canonical_hint:
        return canonical_hint

    text = (query_text or "").strip()
    if not text:
        return "all"

    if len(text) <= 12:
        return "bm25+system"

    semantic_cues = ["如何", "怎么", "原因", "为什么", "机制", "导致", "影响", "诊断", "排查"]
    if any(cue in text for cue in semantic_cues):
        return "simkgc+system"

    return "all"


def _expand_route_to_engines(route: str) -> list[str]:
    valid = ["bm25", "simkgc", "system"]
    normalized = (route or "").strip().lower()

    if normalized == "hybrid":
        return ["bm25", "simkgc"]
    if normalized == "all":
        return valid

    tokens = [token.strip() for token in normalized.replace(",", "+").split("+") if token.strip()]
    selected: list[str] = []
    for token in tokens:
        if token in valid and token not in selected:
            selected.append(token)

    if not selected:
        return ["system"]

    return selected


@tool(
    name_or_callable="agentic_rag_search",
    description=(
        "Agentic RAG 路由工具：先做意图判定，再按状态机强制执行 bm25 / SimKGC / 系统 RAG，"
        "可按单引擎或组合方式运行。状态跳转：analyze_intent -> route -> execute -> done。"
    ),
    args_schema=AgenticRagSearchInput,
)
async def agentic_rag_search(
    query_text: str,
    route_hint: str | None = None,
    simkgc_chroma_dir: str = "chroma_db_full",
    simkgc_l1_collection: str = "simkgc_power_l1_full",
    simkgc_l0_collection: str = "simkgc_power_l0_full",
    simkgc_text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
    system_db_names: list[str] | None = None,
) -> dict[str, Any]:
    state_path = ["analyze_intent"]
    route = _decide_agentic_rag_route(query_text=query_text, route_hint=route_hint)
    state_path.append(f"route:{route}")

    engines = _expand_route_to_engines(route)
    state_path.append(f"execute:{'+'.join(engines)}")

    tasks: dict[str, asyncio.Task] = {}
    if "bm25" in engines:
        tasks["bm25"] = asyncio.create_task(_run_bm25_search_stub(query_text))
    if "simkgc" in engines:
        tasks["simkgc"] = asyncio.create_task(
            _run_simkgc_pg_search(
                query_text=query_text,
                chroma_dir=simkgc_chroma_dir,
                l1_collection=simkgc_l1_collection,
                l0_collection=simkgc_l0_collection,
                text_encoder=simkgc_text_encoder,
            )
        )
    if "system" in engines:
        tasks["system"] = asyncio.create_task(
            _run_system_rag_search(query_text=query_text, db_names=system_db_names)
        )

    engine_results: dict[str, Any] = {"bm25": None, "simkgc": None, "system": None}
    for name, task in tasks.items():
        try:
            engine_results[name] = await task
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.error(f"agentic_rag_search {name} 执行失败: {exc}")
            engine_results[name] = {"ok": False, "engine": name, "message": str(exc)}

    state_path.append("done")
    return {
        "route": route,
        "state_path": state_path,
        "bm25": engine_results["bm25"],
        "simkgc": engine_results["simkgc"],
        "system": engine_results["system"],
    }

def get_kb_based_tools(db_names: list[str] | None = None) -> list:
    """获取所有知识库基于的工具"""
    # 获取所有知识库
    kb_tools = []
    retrievers = knowledge_base.get_retrievers()
    if db_names is None:
        db_ids = None
    else:
        db_ids = [kb_id for kb_id, kb in retrievers.items() if kb["name"] in db_names]

    def _create_retriever_wrapper(db_id: str, retriever_info: dict[str, Any]):
        """创建检索器包装函数的工厂函数，避免闭包变量捕获问题"""

        async def async_retriever_wrapper(
            query_text: str, operation: str = "search", file_name: str | None = None
        ) -> Any:
            """异步检索器包装函数，支持检索和获取思维导图"""

            # 获取思维导图
            if operation == "get_mindmap":
                try:
                    logger.debug(f"Getting mindmap for database {db_id}")

                    from src.repositories.knowledge_base_repository import KnowledgeBaseRepository

                    kb_repo = KnowledgeBaseRepository()
                    kb = await kb_repo.get_by_id(db_id)

                    if kb is None:
                        return f"知识库 {retriever_info['name']} 不存在"

                    mindmap_data = kb.mindmap

                    if not mindmap_data:
                        return f"知识库 {retriever_info['name']} 还没有生成思维导图。"

                    # 将思维导图数据转换为文本格式，便于AI理解
                    def mindmap_to_text(node, level=0):
                        """递归将思维导图JSON转换为层级文本"""
                        indent = "  " * level
                        text = f"{indent}- {node.get('content', '')}\n"
                        for child in node.get("children", []):
                            text += mindmap_to_text(child, level + 1)
                        return text

                    mindmap_text = f"知识库 {retriever_info['name']} 的思维导图结构：\n\n"
                    mindmap_text += mindmap_to_text(mindmap_data)

                    logger.debug(f"Successfully retrieved mindmap for {db_id}")
                    return mindmap_text

                except Exception as e:
                    logger.error(f"Error getting mindmap for {db_id}: {e}")
                    return f"获取思维导图失败: {str(e)}"

            # 默认：检索知识库
            retriever = retriever_info["retriever"]
            try:
                logger.debug(f"Retrieving from database {db_id} with query: {query_text}")
                kwargs = {}
                if file_name:
                    kwargs["file_name"] = file_name

                if asyncio.iscoroutinefunction(retriever):
                    result = await retriever(query_text, **kwargs)
                else:
                    result = retriever(query_text, **kwargs)
                logger.debug(f"Retrieved {len(result) if isinstance(result, list) else 'N/A'} results from {db_id}")
                return result
            except Exception as e:
                logger.error(f"Error in retriever {db_id}: {e}")
                return f"检索失败: {str(e)}"

        return async_retriever_wrapper

    for db_id, retrieve_info in retrievers.items():
        if db_ids is not None and db_id not in db_ids:
            continue

        try:
            # 构建工具描述
            description = (
                f"使用 {retrieve_info['name']} 知识库的多功能工具。\n"
                f"知识库描述：{retrieve_info['description'] or '没有描述。'}\n\n"
                f"支持的操作：\n"
                f"1. 'search' - 检索知识库内容：根据关键词查询相关文档片段\n"
                f"2. 'get_mindmap' - 获取思维导图：查看知识库的整体结构和文件分类\n\n"
                f"使用建议：\n"
                f"- 需要查询具体内容时，使用 operation='search'\n"
                f"- 想了解知识库结构、文件分类时，使用 operation='get_mindmap'"
            )

            # 使用工厂函数创建检索器包装函数，避免闭包问题
            retriever_wrapper = _create_retriever_wrapper(db_id, retrieve_info)

            safename = retrieve_info["name"].replace(" ", "_")[:20]

            args_schema = KnowledgeRetrieverModel
            if retrieve_info["metadata"]["kb_type"] in ["milvus"]:
                args_schema = CommonKnowledgeRetriever

            # 使用 StructuredTool.from_function 创建异步工具
            tool = StructuredTool.from_function(
                coroutine=retriever_wrapper,
                name=safename,
                description=description,
                args_schema=args_schema,
                metadata=retrieve_info["metadata"] | {"tag": ["knowledgebase"]},
            )

            kb_tools.append(tool)
            # logger.debug(f"Successfully created tool {tool_id} for database {db_id}")

        except Exception as e:
            logger.error(f"Failed to create tool for database {db_id}: {e}, \n{traceback.format_exc()}")
            continue

    return kb_tools


def gen_tool_info(tools) -> list[dict[str, Any]]:
    """获取所有工具的信息（用于前端展示）"""
    tools_info = []

    try:
        # 获取注册的工具信息
        for tool_obj in tools:
            try:
                metadata = getattr(tool_obj, "metadata", {}) or {}
                info = {
                    "id": tool_obj.name,
                    "name": metadata.get("name", tool_obj.name),
                    "description": tool_obj.description,
                    "metadata": metadata,
                    "args": [],
                    # "is_async": is_async  # Include async information
                }

                if hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
                    if isinstance(tool_obj.args_schema, dict):
                        schema = tool_obj.args_schema
                    else:
                        schema = tool_obj.args_schema.schema()

                    for arg_name, arg_info in schema.get("properties", {}).items():
                        info["args"].append(
                            {
                                "name": arg_name,
                                "type": arg_info.get("type", ""),
                                "description": arg_info.get("description", ""),
                            }
                        )

                tools_info.append(info)
                # logger.debug(f"Successfully processed tool info for {tool_obj.name}")

            except Exception as e:
                logger.error(
                    f"Failed to process tool {getattr(tool_obj, 'name', 'unknown')}: {e}\n{traceback.format_exc()}. "
                    f"Details: {dict(tool_obj.__dict__)}"
                )
                continue

    except Exception as e:
        logger.error(f"Failed to get tools info: {e}\n{traceback.format_exc()}")
        return []

    logger.info(f"Successfully extracted info for {len(tools_info)} tools")
    return tools_info


def get_buildin_tools() -> list:
    """注册静态工具"""
    static_tools = [
        query_knowledge_graph,
        get_approved_user_goal,
        my_custom_tool,
        agentic_rag_search,
        hybrid_rag_search,
        simkgc_pg_search,
        calculator,
        text_to_img_demo,
    ]

    # subagents 工具
    from .subagents import calc_agent_tool

    static_tools.append(calc_agent_tool)

    # 检查是否启用网页搜索（即是否配置了 API_KEY）
    if config.enable_web_search:
        tavily_search = get_tavily_search()
        if tavily_search:
            static_tools.append(tavily_search)

    return static_tools


async def get_tools_from_context(context, extra_tools=None) -> list:
    """从上下文配置中获取工具列表"""
    # 1. 基础工具 (从 context.tools 中筛选)
    all_basic_tools = get_buildin_tools() + (extra_tools or [])
    selected_tools = []

    if context.tools:
        # 创建工具映射表
        tools_map = {t.name: t for t in all_basic_tools}
        for tool_name in context.tools:
            if tool_name in tools_map:
                selected_tools.append(tools_map[tool_name])

    # 2. 知识库工具
    if context.knowledges:
        kb_tools = get_kb_based_tools(db_names=context.knowledges)
        selected_tools.extend(kb_tools)

    # 3. MCP 工具（使用统一入口，自动过滤 disabled_tools）
    if context.mcps:
        for server_name in context.mcps:
            mcp_tools = await get_enabled_mcp_tools(server_name)
            selected_tools.extend(mcp_tools)

    return selected_tools
