from __future__ import annotations

from typing import Any, Callable


def explain_match(
    product: dict[str, Any],
    room_description: str,
    text_query: str,
    llm_callable: Callable[[str], str] | None = None,
) -> str:
    title = str(product.get("title") or "this rug")
    tags = str(product.get("tags") or "")
    prompt = (
        f"Room context: {room_description or 'not specified'}\n"
        f"User preference: {text_query or 'not specified'}\n"
        f"Rug: {title}\n"
        f"Tags: {tags}\n\n"
        "In one sentence, explain why this rug is a good match for this room. "
        "Be specific about color, style, or size. Do not use generic phrases."
    )

    if llm_callable is not None:
        try:
            response = llm_callable(prompt)
            if response:
                return response.strip()
        except Exception:
            pass

    detected_palette = room_description or "the room's palette"
    detected_style = text_query or "the user's preference"
    return f"This rug complements {detected_palette} and aligns with {detected_style} through its color and style cues."


if __name__ == "__main__":
    demo_product = {"title": "Vintage Persian Area Rug 8x10", "tags": "traditional, beige, oriental"}
    print(explain_match(demo_product, "warm neutral living room", "traditional 8x10"))
    print(explain_match(demo_product, "", "", llm_callable=lambda prompt: "It matches the room well.") )
