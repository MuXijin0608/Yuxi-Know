from __future__ import annotations

from typing import TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.skill_repository import SkillRepository
from src.storage.postgres.manager import pg_manager
from src.storage.postgres.models_business import Skill
from src.utils.logging_config import logger


class SkillPromptMetadata(TypedDict):
    name: str
    description: str
    path: str


class SkillDependencyNode(TypedDict):
    tools: list[str]
    mcps: list[str]
    skills: list[str]


class SkillSessionSnapshot(TypedDict):
    selected_skills: list[str]
    visible_skills: list[str]
    prompt_metadata: dict[str, SkillPromptMetadata]
    dependency_map: dict[str, SkillDependencyNode]


def normalize_selected_skills(selected_skills: list[str] | None) -> list[str]:
    return _normalize_string_list(selected_skills)


def is_snapshot_match_selected_skills(
    snapshot: SkillSessionSnapshot | None,
    selected_skills: list[str] | None,
) -> bool:
    if not snapshot:
        return False
    current = snapshot.get("selected_skills")
    if not isinstance(current, list):
        return False
    return current == normalize_selected_skills(selected_skills)


async def resolve_session_snapshot(
    selected_skills: list[str] | None,
    *,
    db: AsyncSession | None = None,
) -> SkillSessionSnapshot:
    normalized_selected = normalize_selected_skills(selected_skills)
    skills = await _list_skills_from_db(db)
    prompt_metadata, dependency_map = _build_maps(skills)
    visible_skills = expand_skill_closure(normalized_selected, dependency_map)
    return {
        "selected_skills": normalized_selected,
        "visible_skills": visible_skills,
        "prompt_metadata": prompt_metadata,
        "dependency_map": dependency_map,
    }


def collect_prompt_metadata(
    snapshot: SkillSessionSnapshot | None,
    slugs: list[str] | None,
) -> list[SkillPromptMetadata]:
    if not snapshot or not slugs:
        return []

    prompt_metadata = snapshot.get("prompt_metadata") or {}
    result: list[SkillPromptMetadata] = []
    seen: set[str] = set()
    for slug in slugs:
        if not isinstance(slug, str):
            continue
        normalized = slug.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        item = prompt_metadata.get(normalized)
        if not item:
            logger.debug(f"Skill slug not found in session snapshot, skip prompt metadata: {normalized}")
            continue
        result.append(dict(item))
    return result


def build_dependency_bundle(
    snapshot: SkillSessionSnapshot | None,
    activated_slugs: list[str] | None,
) -> dict[str, list[str]]:
    if not snapshot:
        return {"tools": [], "mcps": [], "skills": []}

    dependency_map = snapshot.get("dependency_map") or {}
    closure = expand_skill_closure(activated_slugs or [], dependency_map)
    tools: list[str] = []
    mcps: list[str] = []
    seen_tools: set[str] = set()
    seen_mcps: set[str] = set()

    for slug in closure:
        dep = dependency_map.get(slug, {})
        for tool_name in dep.get("tools", []):
            if tool_name in seen_tools:
                continue
            seen_tools.add(tool_name)
            tools.append(tool_name)
        for mcp_name in dep.get("mcps", []):
            if mcp_name in seen_mcps:
                continue
            seen_mcps.add(mcp_name)
            mcps.append(mcp_name)

    return {"tools": tools, "mcps": mcps, "skills": closure}


def expand_skill_closure(
    slugs: list[str] | None,
    dependency_map: dict[str, SkillDependencyNode],
) -> list[str]:
    ordered_roots = _normalize_string_list(slugs)
    if not ordered_roots:
        return []

    result: list[str] = []
    seen: set[str] = set()

    def dfs(slug: str, stack: set[str]) -> None:
        if slug in stack:
            logger.warning(f"Cycle detected in skill dependencies, skip: {' -> '.join([*stack, slug])}")
            return
        if slug in seen:
            return

        node = dependency_map.get(slug)
        if not node:
            logger.warning(f"Skill dependency target not found in DB snapshot, skip: {slug}")
            return

        seen.add(slug)
        result.append(slug)
        next_stack = set(stack)
        next_stack.add(slug)
        for dep in node.get("skills", []):
            dfs(dep, next_stack)

    for root in ordered_roots:
        dfs(root, set())
    return result


async def get_skill_options_from_db(
    *,
    db: AsyncSession | None = None,
) -> list[dict[str, str]]:
    items = await _list_skills_from_db(db)
    return [
        {
            "id": item.slug,
            "name": item.name,
            "description": item.description,
        }
        for item in items
    ]


async def get_skill_slug_set_from_db(
    *,
    db: AsyncSession | None = None,
) -> set[str]:
    items = await _list_skills_from_db(db)
    return {item.slug for item in items}


def _normalize_string_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _build_maps(skills: list[Skill]) -> tuple[dict[str, SkillPromptMetadata], dict[str, SkillDependencyNode]]:
    prompt_metadata: dict[str, SkillPromptMetadata] = {}
    dependency_map: dict[str, SkillDependencyNode] = {}
    for item in skills:
        prompt_metadata[item.slug] = {
            "name": item.name,
            "description": item.description,
            "path": f"/skills/{item.slug}/SKILL.md",
        }
        dependency_map[item.slug] = {
            "tools": _normalize_string_list(item.tool_dependencies or []),
            "mcps": _normalize_string_list(item.mcp_dependencies or []),
            "skills": _normalize_string_list(item.skill_dependencies or []),
        }
    return prompt_metadata, dependency_map


async def _list_skills_from_db(db: AsyncSession | None) -> list[Skill]:
    if db is not None:
        repo = SkillRepository(db)
        return await repo.list_all()

    try:
        async with pg_manager.get_async_session_context() as session:
            repo = SkillRepository(session)
            return await repo.list_all()
    except RuntimeError:
        # 在非 FastAPI 生命周期场景（如 worker/脚本）按需初始化
        pg_manager.initialize()
        async with pg_manager.get_async_session_context() as session:
            repo = SkillRepository(session)
            return await repo.list_all()
