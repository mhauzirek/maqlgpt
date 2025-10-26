#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

"""
GoodData LDM JSON shrinker for LLM usage.

Goals
- Keep what an LLM needs to understand business intent and write MAQL
  - ids for all logical objects
  - dataset graph and cardinality
  - human metadata that adds meaning: title, description if not equal to title, tags if not empty
  - labels preserved as alternate display forms with minimal fields
  - sourceColumnDataType kept only when its value is not "STRING"
- Remove noisy or physical details
  - physical mappings (dataSourceTableId and internals, sourceColumn)
  - reference source columns and datatypes
  - dateInstances granularities and granularitiesFormatting
- Collapse references
  - keep sources only for date links to preserve the date binding
  - for non-date links keep identifier and multivalue
- Tidy metadata
  - drop description if empty or equal to title
  - drop tags that are empty or duplicates or repeat a parent dataset title or tags
  - drop valueType when it equals TEXT
- Minify and print size stats
"""

UNWANTED_KEYS_GLOBAL = {
    "sourceColumn",
    "defaultView",
    "dataSourceTableId",              # drop physical table mapping entirely
    "granularities",                  # dateInstances size saver
    "granularitiesFormatting",        # dateInstances size saver
}

# case insensitive unwanted keys
UNWANTED_KEYS_CI = {"workspacedatafiltercolumns"}

# fields we allow on each type, in the final compact form
DATASET_ALLOW = {"id", "title", "description", "tags", "grain", "attributes", "facts", "references"}
ATTRIBUTE_ALLOW = {"id", "title", "description", "tags", "labels", "sourceColumnDataType"}
LABEL_ALLOW = {"id", "title", "description", "tags", "valueType", "sourceColumnDataType"}
FACT_ALLOW = {"id", "title", "description", "tags", "sourceColumnDataType"}
GRAIN_ITEM_ALLOW = {"id", "type"}
REFERENCE_ALLOW = {"identifier", "multivalue", "sources"}
IDENTIFIER_ALLOW = {"id", "type"}
SOURCE_TARGET_ALLOW = {"id", "type"}

# --------------------------------------------
# generic helpers
# --------------------------------------------

def _norm(s: Optional[str]) -> str:
    return (s or "").strip()

def _dedup_list_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out

def _clean_description(title: Optional[str], description: Optional[str]) -> Optional[str]:
    t = _norm(title)
    d = _norm(description)
    if not d:
        return None
    if t and d == t:
        return None
    return d

def _clean_tags(
    tags: Optional[List[Any]],
    *,
    self_title: Optional[str] = None,
    parent_title: Optional[str] = None,
    parent_tags: Optional[List[str]] = None,
) -> Optional[List[str]]:
    if not isinstance(tags, list):
        return None
    cleaned: List[str] = []
    for x in tags:
        if isinstance(x, str):
            v = x.strip()
            if v:
                cleaned.append(v)
    if not cleaned:
        return None

    # remove tags equal to this object's title
    self_title = _norm(self_title)
    if self_title:
        cleaned = [t for t in cleaned if t != self_title]

    # remove tags that simply repeat parent dataset title or parent dataset tags
    p_title = _norm(parent_title)
    p_tags = set((parent_tags or []))
    if p_title:
        p_tags.add(p_title)

    cleaned = [t for t in cleaned if t not in p_tags]
    if not cleaned:
        return None

    cleaned = _dedup_list_preserve_order(cleaned)
    return cleaned or None

# --------------------------------------------
# first pass: shape aware compaction
# --------------------------------------------

def _compact_identifier(ident: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(ident, dict):
        return None
    out = {}
    if isinstance(ident.get("id"), str):
        out["id"] = ident["id"]
    if isinstance(ident.get("type"), str):
        out["type"] = ident["type"]
    return out or None

def _compact_sources_for_reference(src_list: Any) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
    """
    Returns (is_date_link, compact_sources_or_none).
    Keep only date targets: {"target": {"id": "..", "type": "date"}}
    For non-date links, we will drop sources entirely.
    """
    if not isinstance(src_list, list):
        return (False, None)

    compact = []
    for s in src_list:
        if not isinstance(s, dict):
            continue
        tgt = s.get("target")
        if isinstance(tgt, dict) and tgt.get("type") == "date" and isinstance(tgt.get("id"), str):
            compact.append({"target": {"id": tgt["id"], "type": "date"}})

    if compact:
        return (True, compact)
    return (False, None)

def _compact_references(refs: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(refs, list):
        return None
    out: List[Dict[str, Any]] = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        new_r: Dict[str, Any] = {}

        ident = _compact_identifier(r.get("identifier"))
        if ident:
            new_r["identifier"] = ident

        # keep multivalue if present as boolean
        if isinstance(r.get("multivalue"), bool):
            new_r["multivalue"] = r["multivalue"]

        # keep sources only for date bindings
        is_date_link, compact_sources = _compact_sources_for_reference(r.get("sources"))
        if is_date_link and compact_sources:
            new_r["sources"] = compact_sources

        # only keep allowed keys
        new_r = {k: v for k, v in new_r.items() if k in REFERENCE_ALLOW}
        if new_r:
            out.append(new_r)

    return out or None

def _compact_grain(grain: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(grain, list):
        return None
    out = []
    for g in grain:
        if not isinstance(g, dict):
            continue
        item = {}
        if isinstance(g.get("id"), str):
            item["id"] = g["id"]
        if isinstance(g.get("type"), str):
            item["type"] = g["type"]
        item = {k: v for k, v in item.items() if k in GRAIN_ITEM_ALLOW}
        if item:
            out.append(item)
    return out or None

def _maybe_keep_source_col_dtype(val: Any) -> Optional[str]:
    # keep only if string and not equal to "STRING"
    if isinstance(val, str) and val != "STRING":
        return val
    return None

def _compact_label(label: Any, *, dataset_title: Optional[str], dataset_tags: Optional[List[str]], parent_attr_title: Optional[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(label, dict):
        return None
    title = label.get("title")
    out: Dict[str, Any] = {}
    if isinstance(label.get("id"), str):
        out["id"] = label["id"]

    desc = _clean_description(title, label.get("description"))
    if isinstance(title, str):
        out["title"] = title
    if desc is not None:
        out["description"] = desc

    tags = _clean_tags(label.get("tags"), self_title=title, parent_title=dataset_title, parent_tags=dataset_tags)
    if tags is not None:
        out["tags"] = tags

    # keep valueType unless it is TEXT
    vt = label.get("valueType")
    if isinstance(vt, str) and vt != "TEXT":
        out["valueType"] = vt

    # keep sourceColumnDataType only if not "STRING"
    scdt = _maybe_keep_source_col_dtype(label.get("sourceColumnDataType"))
    if scdt is not None:
        out["sourceColumnDataType"] = scdt

    out = {k: v for k, v in out.items() if k in LABEL_ALLOW}
    if out.get("id"):
        return out
    return None

def _compact_attribute(attr: Any, *, dataset_title: Optional[str], dataset_tags: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    if not isinstance(attr, dict):
        return None
    title = attr.get("title")
    out: Dict[str, Any] = {}
    if isinstance(attr.get("id"), str):
        out["id"] = attr["id"]

    desc = _clean_description(title, attr.get("description"))
    if isinstance(title, str):
        out["title"] = title
    if desc is not None:
        out["description"] = desc

    tags = _clean_tags(attr.get("tags"), self_title=title, parent_title=dataset_title, parent_tags=dataset_tags)
    if tags is not None:
        out["tags"] = tags

    # labels
    labels = attr.get("labels")
    if isinstance(labels, list):
        new_labels = []
        for lab in labels:
            c = _compact_label(lab, dataset_title=dataset_title, dataset_tags=dataset_tags, parent_attr_title=title if isinstance(title, str) else None)
            if c:
                new_labels.append(c)
        if new_labels:
            out["labels"] = new_labels

    # keep sourceColumnDataType only if not "STRING"
    scdt = _maybe_keep_source_col_dtype(attr.get("sourceColumnDataType"))
    if scdt is not None:
        out["sourceColumnDataType"] = scdt

    out = {k: v for k, v in out.items() if k in ATTRIBUTE_ALLOW}
    if out.get("id"):
        return out
    return None

def _compact_fact(fact: Any, *, dataset_title: Optional[str], dataset_tags: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    if not isinstance(fact, dict):
        return None
    title = fact.get("title")
    out: Dict[str, Any] = {}
    if isinstance(fact.get("id"), str):
        out["id"] = fact["id"]

    desc = _clean_description(title, fact.get("description"))
    if isinstance(title, str):
        out["title"] = title
    if desc is not None:
        out["description"] = desc

    tags = _clean_tags(fact.get("tags"), self_title=title, parent_title=dataset_title, parent_tags=dataset_tags)
    if tags is not None:
        out["tags"] = tags

    # keep sourceColumnDataType only if not "STRING"
    scdt = _maybe_keep_source_col_dtype(fact.get("sourceColumnDataType"))
    if scdt is not None:
        out["sourceColumnDataType"] = scdt

    out = {k: v for k, v in out.items() if k in FACT_ALLOW}
    if out.get("id"):
        return out
    return None

def _compact_dataset(ds: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(ds, dict):
        return None

    title = ds.get("title")
    ds_title: Optional[str] = title if isinstance(title, str) else None

    # parent dataset tags before cleaning, to compare children against
    raw_tags = ds.get("tags") if isinstance(ds.get("tags"), list) else []
    ds_tags_cleaned = _clean_tags(raw_tags, self_title=ds_title)

    out: Dict[str, Any] = {}
    if isinstance(ds.get("id"), str):
        out["id"] = ds["id"]
    if ds_title:
        out["title"] = ds_title

    desc = _clean_description(ds_title, ds.get("description"))
    if desc is not None:
        out["description"] = desc

    # dataset tags: keep non-empty, but drop if equal to [title]
    if ds_tags_cleaned is not None:
        out["tags"] = ds_tags_cleaned

    # grain
    grain = _compact_grain(ds.get("grain"))
    if grain:
        out["grain"] = grain

    # attributes
    attrs = ds.get("attributes")
    if isinstance(attrs, list):
        new_attrs = []
        for a in attrs:
            ca = _compact_attribute(a, dataset_title=ds_title, dataset_tags=ds_tags_cleaned or [])
            if ca:
                new_attrs.append(ca)
        if new_attrs:
            out["attributes"] = new_attrs

    # facts
    facts = ds.get("facts")
    if isinstance(facts, list):
        new_facts = []
        for f in facts:
            cf = _compact_fact(f, dataset_title=ds_title, dataset_tags=ds_tags_cleaned or [])
            if cf:
                new_facts.append(cf)
        if new_facts:
            out["facts"] = new_facts

    # references
    refs = _compact_references(ds.get("references"))
    if refs:
        out["references"] = refs

    # keep only allowed keys on dataset
    out = {k: v for k, v in out.items() if k in DATASET_ALLOW}
    if out.get("id"):
        return out
    return None

def _compact_date_instance(di: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(di, dict):
        return None
    out: Dict[str, Any] = {}
    if isinstance(di.get("id"), str):
        out["id"] = di["id"]
    if isinstance(di.get("title"), str) and di["title"].strip():
        out["title"] = di["title"].strip()

    # keep non-empty tags
    tags = _clean_tags(di.get("tags"), self_title=di.get("title"))
    if tags is not None:
        out["tags"] = tags

    # note: granularities and granularitiesFormatting are intentionally omitted
    return out or None

def compact_ldm(ldm: Any) -> Any:
    """
    Shape aware compaction of the LDM section.
    """
    if not isinstance(ldm, dict):
        return ldm

    out: Dict[str, Any] = {}

    # datasets
    ds_list = ldm.get("datasets")
    if isinstance(ds_list, list):
        new_ds = []
        for ds in ds_list:
            c = _compact_dataset(ds)
            if c:
                new_ds.append(c)
        if new_ds:
            out["datasets"] = new_ds

    # dateInstances
    di_list = ldm.get("dateInstances")
    if isinstance(di_list, list):
        new_di = []
        for di in di_list:
            c = _compact_date_instance(di)
            if c:
                new_di.append(c)
        if new_di:
            out["dateInstances"] = new_di

    return out or {}

# --------------------------------------------
# second pass: generic pruning from the original script
# --------------------------------------------

def prune_value(v: Any) -> Any:
    """
    Recursively clean a JSON-like value according to generic rules:
     - remove certain keys globally
     - drop description if empty or same as title
     - drop valueType when it is TEXT
     - drop arrays that are empty
     - remove case-insensitive workspaceDataFilterColumns keys
     - keep sourceColumnDataType only when its value is not "STRING"
    """
    if isinstance(v, list):
        cleaned_list = []
        for item in v:
            cleaned_item = prune_value(item)
            if cleaned_item is not None:
                cleaned_list.append(cleaned_item)
        return cleaned_list

    if isinstance(v, dict):
        title_val = v.get("title")

        cleaned = {}
        for k, val in v.items():
            # skip globally unwanted keys
            if k in UNWANTED_KEYS_GLOBAL:
                continue
            if k.lower() in UNWANTED_KEYS_CI:
                continue

            # keep sourceColumnDataType only if not "STRING"
            if k == "sourceColumnDataType":
                if isinstance(val, str) and val != "STRING":
                    cleaned[k] = val
                # if STRING or not a str we drop it by not adding
                continue

            # remove valueType if it equals TEXT
            if k == "valueType" and isinstance(val, str) and val == "TEXT":
                continue

            # description vs title
            if k == "description":
                if (val == "") or (isinstance(title_val, str) and isinstance(val, str) and val == title_val):
                    continue

            new_val = prune_value(val)

            # remove arrays that are empty
            if isinstance(new_val, list) and len(new_val) == 0:
                continue

            cleaned[k] = new_val

        return cleaned

    return v

# --------------------------------------------
# root handling and IO
# --------------------------------------------

def clean_root(root: Any) -> Any:
    """
    - remove top-level layout if present
    - preserve only the LDM subtree, compact it shape-aware, then apply generic pruning
    """
    if not isinstance(root, dict):
        # if the file is not an object with ldm, still try to prune generically
        return prune_value(root)

    root = dict(root)  # shallow copy
    root.pop("layout", None)

    if "ldm" in root and isinstance(root["ldm"], dict):
        # 1) shape-aware compaction
        compact = compact_ldm(root["ldm"])
        # 2) generic prune to apply global removals and drop empty arrays
        compact = prune_value(compact)
        root = {"ldm": compact}
    else:
        root = prune_value(root)

    return root

def format_bytes(n: int) -> str:
    # human friendly sizes without units beyond bytes and KB and MB
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n/1024:.1f} KB"
    return f"{n/(1024*1024):.2f} MB"

def main():
    parser = argparse.ArgumentParser(description="Minify and reduce GoodData LDM JSON for LLM usage")
    parser.add_argument("input_json", help="Path to the LDM JSON file")
    args = parser.parse_args()

    in_path = args.input_json
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # original size from file bytes
    try:
        original_size = os.path.getsize(in_path)
    except OSError:
        # fallback to in-memory re-serialization
        original_size = len(json.dumps(data))

    cleaned = clean_root(data)

    base, ext = os.path.splitext(in_path)
    out_path = f"{base}_mini.json"

    # minify
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, separators=(",", ":"))

    # output size
    try:
        output_size = os.path.getsize(out_path)
    except OSError:
        output_size = len(json.dumps(cleaned, separators=(",", ":")))

    # ratio
    ratio = (1.0 - (output_size / max(original_size, 1))) * 100.0

    print(f"Wrote {out_path}")
    print(f"Original size: {format_bytes(original_size)}")
    print(f"Output size:   {format_bytes(output_size)}")
    print(f"Reduction:     {ratio:.2f}%")

if __name__ == "__main__":
    main()
