from typing import Any

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from .dom import DOM_ARGS, SNAPSHOT_JS
from .errors import GameStateTimeoutError


async def wait_snapshot(
    page: Page,
    predicate: str,
    *,
    values: dict[str, Any] | None = None,
    timeout_ms: float = 10_000,
    description: str,
) -> None:
    """Wait on the shared DOM snapshot, using developer-authored predicates."""
    expression = (
        "(args) => {const state = ("
        + SNAPSHOT_JS
        + ")(args.dom); const v=args.values; return ("
        + predicate
        + ");}"
    )
    try:
        handle = await page.wait_for_function(
            expression,
            arg={"dom": {**DOM_ARGS, "includeHidden": False}, "values": values or {}},
            timeout=timeout_ms,
        )
        await handle.dispose()
    except PlaywrightTimeoutError as exc:
        raise GameStateTimeoutError(
            f"Timed out waiting for {description}; inspect before retrying"
        ) from exc
