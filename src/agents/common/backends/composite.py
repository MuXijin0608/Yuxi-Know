from deepagents.backends import CompositeBackend, StateBackend

from src.services.skill_resolver import normalize_selected_skills
from src.services.skill_service import is_valid_skill_slug

from .skills_backend import SelectedSkillsReadonlyBackend


def _get_visible_skills_from_runtime(runtime) -> list[str]:
    context = getattr(runtime, "context", None)
    snapshot = getattr(context, "skill_session_snapshot", None)
    if isinstance(snapshot, dict):
        visible = snapshot.get("visible_skills")
        if isinstance(visible, list):
            return [slug for slug in visible if isinstance(slug, str) and is_valid_skill_slug(slug)]

    selected = getattr(context, "skills", None) or []
    return normalize_selected_skills(selected)


def create_agent_composite_backend(runtime) -> CompositeBackend:
    """为 agent 构建 backend：默认 StateBackend + /skills 路由只读 backend。"""
    visible_skills = _get_visible_skills_from_runtime(runtime)
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/skills/": SelectedSkillsReadonlyBackend(selected_slugs=visible_skills),
        },
    )
