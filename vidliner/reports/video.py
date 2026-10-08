"""Offline source/variant video comparison without a server or third-party assets."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from vidliner.core.errors import ErrorCode, ValidationFailure


def render_video_review(manifest: dict[str, Any], output_path: Path) -> str:
    """Render a review page whose local media paths are relative to the HTML file."""
    try:
        if manifest.get("format") != "vidliner/video-render@1":
            raise ValueError("unsupported video manifest format")
        records = manifest["variants"]
        if not isinstance(records, list) or not records:
            raise ValueError("manifest must contain rendered variants")
        variants = []
        source: Path | None = None
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("variant must be an object")
            current_source = Path(record["input"])
            path = Path(record["output"])
            if not current_source.is_absolute() or not path.is_absolute():
                raise ValueError("manifest media paths must be absolute")
            if not current_source.is_file() or not path.is_file():
                raise ValueError("manifest media files are missing")
            if output_path.resolve() in {current_source.resolve(), path.resolve()}:
                raise ValueError("preview must not overwrite media")
            if not isinstance(record.get("variant_id"), str):
                raise ValueError("variant_id must be a string")
            if source is not None and current_source != source:
                raise ValueError("all variants must share one source")
            source = current_source
            variants.append(
                {**record, "url": quote(os.path.relpath(path, output_path.parent.resolve()), safe="/")}
            )
        assert source is not None
        source_url = quote(os.path.relpath(source, output_path.parent.resolve()), safe="/")
        payload = json.dumps(variants, ensure_ascii=False, allow_nan=False)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ValidationFailure(
            f"invalid video review manifest: {exc}", code=ErrorCode.EXPORT_INVALID
        ) from exc
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return _TEMPLATE.replace("__SOURCE_URL__", html.escape(source_url, quote=True)).replace(
        "__VARIANTS__", payload
    )


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VidLiner · Video review</title><style>
:root{color-scheme:dark;font:14px system-ui,sans-serif;background:#0c1423;color:#e3ebf7}body{margin:0;padding:24px}
header{display:flex;gap:20px;align-items:center}header strong{color:#57dbc9;letter-spacing:2px}h1{font-size:20px}
.players{display:grid;grid-template-columns:1fr 1fr;gap:20px}video{width:100%;height:45vh;background:#060b13;object-fit:contain}
label{display:block;margin:12px 0 8px;color:#98abc8}select{max-width:100%;padding:8px;background:#162238;color:inherit;border:1px solid #435a7c}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#111c2e;padding:20px;font:12px/1.6 ui-monospace,monospace}
.hint{color:#98abc8}a{color:#57dbc9}@media(max-width:760px){.players{grid-template-columns:1fr}video{height:30vh}}
</style></head><body><header><strong>VIDLINER</strong><h1>Video review</h1></header>
<p class="hint">Local render · labels and acceptance remain unverified</p>
<div class="players"><section><label for="source">Source</label><video id="source" controls preload="metadata" src="__SOURCE_URL__"></video></section>
<section><label for="variant">Rendered variant</label><video id="variant" controls preload="metadata"></video></section></div>
<label for="selection">Choose variant</label><select id="selection"></select><p><a id="media">Open rendered video</a></p><pre id="evidence"></pre>
<script type="application/json" id="variants">__VARIANTS__</script><script>
const variants=JSON.parse(document.getElementById('variants').textContent), selection=document.getElementById('selection');
variants.forEach((variant,index)=>{const option=document.createElement('option');option.value=index;option.textContent=(index+1)+' · '+variant.variant_id;selection.appendChild(option);});
function show(){const variant=variants[Number(selection.value)];document.getElementById('variant').src=variant.url;document.getElementById('media').href=variant.url;
const {url,...evidence}=variant;document.getElementById('evidence').textContent=JSON.stringify(evidence,null,2);}
selection.addEventListener('change',show);show();
</script></body></html>"""
