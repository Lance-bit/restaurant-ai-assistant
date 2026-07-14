"""
Document ingestion + chunking.

Chunking strategy: one semantic unit per chunk (one menu item = one chunk,
one policy clause = one chunk) rather than fixed-size/character-window
chunking. Justification:
  - Fixed-size windows risk splitting a dish's name from its allergen list,
    or a policy's condition from its exception -- exactly the kind of split
    that causes hallucination or wrong-but-plausible answers ("is it vegan?"
    answered from a chunk that no longer contains the dietary_tags field).
  - Each menu item / policy clause is already a complete, self-contained
    fact in the source JSON, so chunking at that boundary preserves full
    context in every retrieved chunk -- no need for a chunk-overlap window.
  - Keeps chunks small and topically pure, which improves retrieval
    precision (top-k results are less likely to be "half relevant").
"""
from typing import List, Dict, Any
import json


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_menu(menu_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []
    for item in menu_json["items"]:
        allergens = ", ".join(item["allergens"]) if item["allergens"] else "none"
        dietary = ", ".join(item["dietary_tags"]) if item["dietary_tags"] else "none"
        text = (
            f"{item['name']}: {item['description']} "
            f"Ingredients: {', '.join(item['ingredients'])}. "
            f"Cooking method: {item['cooking_method']}. "
            f"Dietary tags: {dietary}. Allergens: {allergens}. "
            f"Price: {item['price']} EGP. "
            f"Available at: {', '.join(item['branches'])}."
        )
        chunks.append({
            "text": text,
            "source": "menu",
            "item_name": item["name"],
            "branches": item["branches"],
        })
    return chunks


def chunk_policies(policies_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []
    for section in policies_json["sections"]:
        text = f"{section['title']}: {section['content']}"
        chunks.append({
            "text": text,
            "source": "policy",
            "section": section["title"],
            "branches": section.get("branches", ["all"]),
        })
    return chunks


def get_all_menu_item_names(menu_json: Dict[str, Any]) -> List[str]:
    """Used by the groundedness checker to catch invented dish names."""
    return [item["name"] for item in menu_json["items"]]
