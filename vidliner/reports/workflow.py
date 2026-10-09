"""Self-contained workflow layout and configuration editor with no network dependencies."""

from __future__ import annotations

import json
from importlib.resources import files

from vidliner.domain.workflow import WorkflowDocument

SUPPORTED_LOCALES = ("en-US", "zh-CN", "ja-JP", "ko-KR", "es-ES")


def render_workflow(document: WorkflowDocument, *, host_config: dict | None = None) -> str:
    """Embed inert JSON; all document strings enter the DOM through textContent."""
    payload = (
        document.model_dump_json().replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )
    web_files = files("vidliner.web")
    locales = {
        locale: json.loads(web_files.joinpath("locales", f"{locale}.json").read_text(encoding="utf-8"))
        for locale in SUPPORTED_LOCALES
    }
    locale_payload = (
        json.dumps(locales, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    i18n_script = web_files.joinpath("i18n.js").read_text(encoding="utf-8")
    html = (
        _TEMPLATE.replace("__WORKFLOW_JSON__", payload)
        .replace("__LOCALES_JSON__", locale_payload)
        .replace("__I18N_SCRIPT__", i18n_script)
    )
    if host_config is not None:
        boot = json.dumps(host_config).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
        script = web_files.joinpath("canvas.js").read_text(encoding="utf-8")
        github_script = web_files.joinpath("github.js").read_text(encoding="utf-8")
        style = web_files.joinpath("canvas.css").read_text(encoding="utf-8")
        html = html.replace(
            "</body>",
            f'<style>{style}</style><script type="application/json" id="host">{boot}</script><script>{script}</script><script>{github_script}</script></body>',
        )
    return html


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VidLiner · Workflow</title>
<style>
:root{color-scheme:light;font:14px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f3f0e8;color:#181b24;--ink:#181b24;--paper:#fffdf7;--sand:#ece8dd;--line:#d8d2c5;--muted:#6e7b70;--route:#9848d8;--signal:#f1c84b;--quiet:#a8b5aa}
*{box-sizing:border-box}body{margin:0;min-width:320px}header{height:72px;display:flex;align-items:center;gap:18px;padding:0 28px;background:var(--ink);color:var(--paper);border-bottom:4px solid var(--signal)}
header strong{font-size:13px;letter-spacing:3px;color:var(--paper)}h1{font-size:16px;font-weight:500;margin:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}header span{margin-left:auto;color:var(--quiet);font:12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.04em}
.toolbar{display:flex;gap:8px;padding:12px 24px;align-items:center;flex-wrap:wrap;background:var(--paper);border-bottom:1px solid var(--line)}
input,button,textarea,select{font:inherit;color:var(--ink);background:var(--paper);border:1px solid #bdb6a9;border-radius:2px;padding:9px 12px}textarea{width:100%;min-height:180px;resize:vertical;font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}#configEditor[hidden]{display:none}#configError{color:#9c3f3f}
button{cursor:pointer}button:hover{border-color:var(--route)}button:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible{outline:3px solid rgba(152,72,216,.28);outline-offset:2px}input{width:230px}main{display:grid;grid-template-columns:minmax(0,1fr) 360px;height:calc(100vh - 145px);min-height:420px}
svg{width:100%;height:100%;touch-action:none;cursor:grab;background:var(--paper)}
svg:active{cursor:grabbing}aside{overflow:auto;background:var(--sand);border-left:1px solid var(--line);padding:24px}h2{font-size:18px;line-height:1.2;margin:0 0 5px;color:var(--ink)}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:#3f4940;background:rgba(255,253,247,.72);border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:14px;margin:16px 0 0}
.hint{color:var(--muted);font-size:12px}.node{cursor:pointer}.node:focus{outline:none}.node:focus rect{stroke:var(--signal)}.node.selected rect{stroke:var(--route);stroke-width:3}.edge{fill:none;stroke:#8f9b8f;stroke-width:2}.muted{opacity:.18}.editing .node{cursor:move}.editing .node rect{stroke-dasharray:5 4}.editing #edit{border-color:var(--route);color:var(--route)}
@media(max-width:760px){header{height:64px;padding:0 16px}header span{display:none}.toolbar{padding:10px 14px}.toolbar input{flex:1;min-width:180px}main{grid-template-columns:minmax(0,1fr);height:auto}svg{height:58vh;min-height:360px}aside{border-left:0;border-top:1px solid var(--line);padding:20px}.toolbar button{min-height:42px}}
</style></head><body>
<header><strong>VIDLINER</strong><h1 id="name"></h1><span id="counts"></span></header>
<div class="toolbar"><input id="search" data-i18n-label="toolbar.searchLabel" data-i18n-placeholder="toolbar.search">
<button id="fit" data-i18n="toolbar.fit"></button><button id="zoomIn" data-i18n="toolbar.zoomIn" data-i18n-label="toolbar.zoomIn">+</button><button id="zoomOut" data-i18n="toolbar.zoomOut" data-i18n-label="toolbar.zoomOut">-</button>
<button id="edit" data-i18n="toolbar.edit"></button><button id="download" data-i18n="toolbar.save"></button>
<span class="hint" id="status" data-i18n="status.planSnapshot"></span><label class="locale-label" for="localeSelect" data-i18n="toolbar.language"></label><select id="localeSelect" aria-label="Language"><option value="en-US">EN</option><option value="zh-CN">中文</option><option value="ja-JP">日本語</option><option value="ko-KR">한국어</option><option value="es-ES">Español</option></select></div>
<main><svg id="canvas" data-i18n-label="toolbar.routeMap" aria-label="Workflow route map"><defs><pattern id="grid" width="64" height="64" patternUnits="userSpaceOnUse"><path d="M64 0H0V64" fill="none" stroke="#d8d2c5" stroke-opacity=".56"/></pattern></defs><rect width="100%" height="100%" fill="url(#grid)"/><g id="scene"></g></svg>
<aside><h2 id="selection" data-i18n="aside.workflow"></h2><p class="hint" data-i18n="aside.notExecuted"></p>
<section id="configEditor" hidden><label for="config" data-i18n="aside.configuration"></label><textarea id="config" spellcheck="false"></textarea>
<button id="applyConfig" data-i18n="aside.applyConfiguration"></button><button id="resetConfig" data-i18n="aside.reset"></button><p id="configError" role="alert"></p></section>
<pre id="detail"></pre></aside></main>
<script type="application/json" id="locales">__LOCALES_JSON__</script><script>__I18N_SCRIPT__</script>
<script type="application/json" id="workflow">__WORKFLOW_JSON__</script>
<script>
const doc=JSON.parse(document.getElementById('workflow').textContent), svg=document.getElementById('canvas'), scene=document.getElementById('scene');
document.title=doc.name+' · VidLiner'; document.getElementById('name').textContent=doc.name;
const i18n=window.VidLinerI18n;
i18n.init();
function updateSummary(){document.getElementById('counts').textContent=i18n.t('counts',{nodes:new Intl.NumberFormat(i18n.locale()).format(doc.nodes.length),connections:new Intl.NumberFormat(i18n.locale()).format(doc.edges.length)});}
updateSummary();
document.getElementById('detail').textContent=JSON.stringify({recipe_hash:doc.recipe_hash,capability_bindings:doc.capability_bindings,unmet_capabilities:doc.unmet_capabilities},null,2);
const ns='http://www.w3.org/2000/svg', index=new Map(doc.nodes.map(n=>[n.id,n])), groups=new Map(), paths=[];let selected=null;
let statusKey='status.planSnapshot';
function el(tag,attrs,parent,text){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;parent.appendChild(e);return e;}
const palette={generate:'#9848d8',verify:'#f1c84b',evaluate:'#7b8b7d',export:'#181b24',plan:'#6e7b70'};
function edgePath(edge){const a=index.get(edge.source).position,b=index.get(edge.target).position;return `M ${a.x+256} ${a.y+48} C ${a.x+296} ${a.y+48}, ${b.x-40} ${b.y+48}, ${b.x} ${b.y+48}`;}
for(const e of doc.edges){
const path=el('path',{d:edgePath(e),class:'edge'},scene);
el('title',{},path,e.source_port+' → '+e.target_port);paths.push({element:path,edge:e});}
function select(n){selected=n;document.getElementById('configEditor').hidden=false;document.getElementById('config').value=JSON.stringify(n.config,null,2);document.getElementById('configError').textContent='';for(const g of groups.values())g.classList.remove('selected');groups.get(n.id).classList.add('selected');
document.getElementById('selection').textContent=n.operator; document.getElementById('detail').textContent=JSON.stringify({
id:n.id,stage:n.stage,version:n.operator_version,description:n.description,inputs:n.inputs,
connections:doc.edges.filter(e=>e.target===n.id),outputs:n.outputs,config:n.config,capabilities:n.needs,
backend_hints:Object.fromEntries(n.needs.map(c=>[c,doc.capability_bindings[c]??'unbound'])),
retry:n.retry,max_parallelism:n.max_parallelism,timeout_s:n.timeout_s,lineage:n.lineage,selects:n.selects},null,2);}
for(const n of doc.nodes){const g=el('g',{transform:`translate(${n.position.x},${n.position.y})`,class:'node',tabindex:'0',role:'button','aria-label':n.operator},scene);
el('rect',{width:256,height:96,rx:2,fill:'#fffdf7',stroke:palette[n.stage]??'#9aa69a','stroke-width':2},g);
el('text',{x:16,y:25,fill:palette[n.stage]??'#6e7b70','font-size':11,'data-stage':n.stage},g,i18n.t('stage.'+n.stage));
el('text',{x:16,y:50,fill:'#181b24','font-size':14},g,n.operator.length>29?n.operator.slice(0,28)+'…':n.operator);
el('text',{x:16,y:75,fill:'#6e7b70','font-size':11},g,n.id.length>32?n.id.slice(0,31)+'…':n.id);
el('title',{},g,n.description||n.operator);g.addEventListener('click',()=>select(n));g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(n);}});groups.set(n.id,g);
let nodeDrag=null;
g.addEventListener('pointerdown',e=>{if(!document.body.classList.contains('editing')||e.button!==0)return;e.stopPropagation();g.setPointerCapture(e.pointerId);g.dataset.dragging='1';nodeDrag={x:e.clientX,y:e.clientY,px:n.position.x,py:n.position.y};});
g.addEventListener('pointermove',e=>{if(g.dataset.dragging!=='1'||!nodeDrag)return;n.position.x=nodeDrag.px+(e.clientX-nodeDrag.x)/scale;n.position.y=nodeDrag.py+(e.clientY-nodeDrag.y)/scale;g.setAttribute('transform',`translate(${n.position.x},${n.position.y})`);for(const p of paths){if(p.edge.source===n.id||p.edge.target===n.id)p.element.setAttribute('d',edgePath(p.edge));}});
g.addEventListener('pointerup',()=>{delete g.dataset.dragging;});g.addEventListener('pointercancel',()=>{delete g.dataset.dragging;});}
let ox=0,oy=0,scale=1;function draw(){scene.setAttribute('transform',`translate(${ox},${oy}) scale(${scale})`);}
function fit(){const bounds=scene.getBBox(),rect=svg.getBoundingClientRect();scale=Math.max(.02,Math.min(1.2,(rect.width-64)/Math.max(bounds.width,1),(rect.height-64)/Math.max(bounds.height,1)));
ox=(rect.width-bounds.width*scale)/2-bounds.x*scale;oy=(rect.height-bounds.height*scale)/2-bounds.y*scale;draw();}
function zoom(factor,x,y){const next=Math.min(4,Math.max(.02,scale*factor)),ratio=next/scale;ox=x-(x-ox)*ratio;oy=y-(y-oy)*ratio;scale=next;draw();}
document.getElementById('fit').onclick=fit;document.getElementById('zoomIn').onclick=()=>zoom(1.25,svg.clientWidth/2,svg.clientHeight/2);
document.getElementById('zoomOut').onclick=()=>zoom(.8,svg.clientWidth/2,svg.clientHeight/2);
document.getElementById('edit').onclick=()=>{document.body.classList.toggle('editing');const active=document.body.classList.contains('editing');statusKey=active?'status.editMode':'status.planSnapshot';document.getElementById('edit').textContent=i18n.t(active?'toolbar.finish':'toolbar.edit');document.getElementById('status').textContent=i18n.t(statusKey);};
document.getElementById('download').onclick=()=>{const blob=new Blob([JSON.stringify(doc,null,2)+'\n'],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='vidliner-workflow.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);};
document.getElementById('resetConfig').onclick=()=>{if(selected)select(selected);};
document.getElementById('applyConfig').onclick=()=>{if(!selected)return;try{const value=JSON.parse(document.getElementById('config').value);if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Configuration must be a JSON object');
selected.config=value;doc.recipe_hash=null;doc.capability_bindings={};doc.unmet_capabilities=[];select(selected);statusKey='status.modifiedDraft';document.getElementById('status').textContent=i18n.t(statusKey);}catch(error){document.getElementById('configError').textContent=error.message||i18n.t('error.unknown');}};
svg.addEventListener('wheel',e=>{e.preventDefault();const r=svg.getBoundingClientRect();zoom(e.deltaY<0?1.1:1/1.1,e.clientX-r.left,e.clientY-r.top);},{passive:false});
let drag=null;svg.addEventListener('pointerdown',e=>{if(e.target.closest('.node')||e.button!==0)return;drag={x:e.clientX,y:e.clientY,ox,oy};svg.setPointerCapture(e.pointerId);});
svg.addEventListener('pointermove',e=>{if(drag){ox=drag.ox+e.clientX-drag.x;oy=drag.oy+e.clientY-drag.y;draw();}});
svg.addEventListener('pointerup',()=>drag=null);svg.addEventListener('pointercancel',()=>drag=null);
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase(),matches=new Set();
for(const n of doc.nodes){const match=(n.operator+' '+n.id+' '+n.lineage.join('/')).toLowerCase().includes(q);groups.get(n.id).classList.toggle('muted',!match);if(match)matches.add(n.id);}
for(const p of paths)p.element.classList.toggle('muted',!matches.has(p.edge.source)||!matches.has(p.edge.target));});
window.addEventListener('resize',fit);fit();
window.addEventListener('vidliner:locale-change',()=>{updateSummary();i18n.apply();for(const stage of scene.querySelectorAll('[data-stage]'))stage.textContent=i18n.t('stage.'+stage.dataset.stage);const active=document.body.classList.contains('editing');document.getElementById('edit').textContent=i18n.t(active?'toolbar.finish':'toolbar.edit');document.getElementById('selection').textContent=selected?selected.operator:i18n.t('aside.workflow');if(!document.getElementById('host'))document.getElementById('status').textContent=i18n.t(statusKey);});
</script></body></html>"""
