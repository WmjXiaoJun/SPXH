"""M9 报告层：把管线证据变成人话（LLM 可选，离线确定性回退）."""
from spxh.core.report.llm_config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMConfig,
    list_models,
    load_config,
    masked_config,
    save_config,
    test_connection,
)
from spxh.core.report.narrator import (
    build_prompt,
    call_llm,
    check_numeric_grounding,
    collect_evidence,
    collect_evidence_numbers,
    llm_available,
    llm_info,
    narrate,
    render_offline,
)

__all__ = [
    "collect_evidence",
    "render_offline",
    "build_prompt",
    "call_llm",
    "llm_available",
    "llm_info",
    "check_numeric_grounding",
    "collect_evidence_numbers",
    "narrate",
    "LLMConfig",
    "load_config",
    "save_config",
    "masked_config",
    "list_models",
    "test_connection",
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
]
