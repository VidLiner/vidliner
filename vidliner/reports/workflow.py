"""Self-contained workflow layout and configuration editor with no network dependencies."""

from __future__ import annotations

import json
from importlib.resources import files

from vidliner.domain.workflow import WorkflowDocument


def render_workflow(document: WorkflowDocument, *, host_config: dict | None = None) -> str:
    """Embed inert JSON; all document strings enter the DOM through textContent."""
    payload = (
        document.model_dump_json().replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )
    html = _TEMPLATE.replace("__WORKFLOW_JSON__", payload)
    if host_config is not None:
        boot = json.dumps(host_config).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
        script = files("vidliner.web").joinpath("canvas.js").read_text(encoding="utf-8")
        style = files("vidliner.web").joinpath("canvas.css").read_text(encoding="utf-8")
        html = html.replace(
            "</body>",
            f'<style>{style}</style><script type="application/json" id="host">{boot}</script><script>{script}</script></body>',
        )
    return html


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VidLiner · Workflow</title>
<style>
:root{color-scheme:dark;font:14px system-ui,sans-serif;background:#0c1423;color:#e3ebf7}
*{box-sizing:border-box}body{margin:0}header{display:flex;align-items:center;gap:18px;padding:18px 24px;border-bottom:1px solid #29354a}
header strong{color:#57dbc9;letter-spacing:2px}h1{font-size:17px;margin:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
header span{margin-left:auto;color:#98abc8}.toolbar{display:flex;gap:8px;padding:12px 24px;align-items:center;flex-wrap:wrap}
input,button,textarea{font:inherit;color:inherit;background:#162238;border:1px solid #354863;border-radius:7px;padding:8px 12px}
textarea{width:100%;min-height:180px;resize:vertical;font:12px/1.6 ui-monospace,monospace}#configEditor[hidden]{display:none}#configError{color:#f2aeae}
button{cursor:pointer}button:hover{border-color:#57dbc9}input{width:230px}main{display:grid;grid-template-columns:minmax(0,1fr) 340px;height:calc(100vh - 145px);min-height:320px}
svg{width:100%;height:100%;touch-action:none;cursor:grab;background:radial-gradient(#263348 1px,transparent 1px);background-size:22px 22px}
svg:active{cursor:grabbing}aside{overflow:auto;background:#111c2e;border-left:1px solid #29354a;padding:20px}
h2{font-size:16px;color:#57dbc9}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace;color:#b9cbe4}
.hint{color:#98abc8;font-size:12px}.node{cursor:pointer}.node:focus{outline:none}.node:focus rect{stroke:#fff}
.node.selected rect{stroke:#fff;stroke-width:3}.edge{fill:none;stroke:#435a7c;stroke-width:1.5}.muted{opacity:.15}
.editing .node{cursor:move}.editing .node rect{stroke-dasharray:4 3}.editing #edit{border-color:#57dbc9;color:#57dbc9}
@media(max-width:760px){main{grid-template-columns:minmax(0,1fr);height:auto}svg{height:55vh;min-height:320px}aside{border-left:0;border-top:1px solid #29354a}header span{display:none}}
</style></head><body>
<header><strong>VIDLINER</strong><h1 id="name"></h1><span id="counts"></span></header>
<div class="toolbar"><input id="search" aria-label="Search operators" placeholder="Search operators or lineage…">
<button id="fit">Fit graph</button><button id="zoomIn" aria-label="Zoom in">+</button><button id="zoomOut" aria-label="Zoom out">-</button>
<button id="edit">Edit layout</button><button id="download">Save JSON</button>
<span class="hint" id="status">Plan snapshot · select a node for ports and config</span></div>
<main><svg id="canvas" aria-label="Workflow graph"><g id="scene"></g></svg>
<aside><h2 id="selection">Workflow</h2><p class="hint">Not executed · backend bindings unverified</p>
<section id="configEditor" hidden><label for="config">Configuration</label><textarea id="config" spellcheck="false"></textarea>
<button id="applyConfig">Apply configuration</button><button id="resetConfig">Reset</button><p id="configError" role="alert"></p></section>
<pre id="detail"></pre></aside></main>
<script type="application/json" id="workflow">__WORKFLOW_JSON__</script>
<script>
const doc=JSON.parse(document.getElementById('workflow').textContent), svg=document.getElementById('canvas'), scene=document.getElementById('scene');
document.title=doc.name+' · VidLiner'; document.getElementById('name').textContent=doc.name;
document.getElementById('counts').textContent=doc.nodes.length+' nodes · '+doc.edges.length+' connections';
document.getElementById('detail').textContent=JSON.stringify({recipe_hash:doc.recipe_hash,capability_bindings:doc.capability_bindings,unmet_capabilities:doc.unmet_capabilities},null,2);
const ns='http://www.w3.org/2000/svg', index=new Map(doc.nodes.map(n=>[n.id,n])), groups=new Map(), paths=[];let selected=null;
function el(tag,attrs,parent,text){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;parent.appendChild(e);return e;}
const palette={generate:'#e8b777',verify:'#57dbc9',evaluate:'#57dbc9',export:'#a99aff',plan:'#7fa9ed'};
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
el('rect',{width:256,height:96,rx:8,fill:'#152338',stroke:palette[n.stage]??'#6584b0','stroke-width':1.5},g);
el('text',{x:16,y:25,fill:palette[n.stage]??'#98abc8','font-size':11},g,n.stage.toUpperCase());
el('text',{x:16,y:50,fill:'#e3ebf7','font-size':14},g,n.operator.length>29?n.operator.slice(0,28)+'…':n.operator);
el('text',{x:16,y:75,fill:'#98abc8','font-size':11},g,n.id.length>32?n.id.slice(0,31)+'…':n.id);
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
document.getElementById('edit').onclick=()=>{document.body.classList.toggle('editing');const active=document.body.classList.contains('editing');document.getElementById('edit').textContent=active?'Finish layout':'Edit layout';document.getElementById('status').textContent=active?'Edit mode · drag nodes, then save JSON':'Plan snapshot · select a node for ports and config';};
document.getElementById('download').onclick=()=>{const blob=new Blob([JSON.stringify(doc,null,2)+'\n'],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='vidliner-workflow.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);};
document.getElementById('resetConfig').onclick=()=>{if(selected)select(selected);};
document.getElementById('applyConfig').onclick=()=>{if(!selected)return;try{const value=JSON.parse(document.getElementById('config').value);if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Configuration must be a JSON object');
selected.config=value;doc.recipe_hash=null;doc.capability_bindings={};doc.unmet_capabilities=[];select(selected);document.getElementById('status').textContent='Modified draft · validation pending';}catch(error){document.getElementById('configError').textContent=error.message;}};
svg.addEventListener('wheel',e=>{e.preventDefault();const r=svg.getBoundingClientRect();zoom(e.deltaY<0?1.1:1/1.1,e.clientX-r.left,e.clientY-r.top);},{passive:false});
let drag=null;svg.addEventListener('pointerdown',e=>{if(e.target.closest('.node')||e.button!==0)return;drag={x:e.clientX,y:e.clientY,ox,oy};svg.setPointerCapture(e.pointerId);});
svg.addEventListener('pointermove',e=>{if(drag){ox=drag.ox+e.clientX-drag.x;oy=drag.oy+e.clientY-drag.y;draw();}});
svg.addEventListener('pointerup',()=>drag=null);svg.addEventListener('pointercancel',()=>drag=null);
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase(),matches=new Set();
for(const n of doc.nodes){const match=(n.operator+' '+n.id+' '+n.lineage.join('/')).toLowerCase().includes(q);groups.get(n.id).classList.toggle('muted',!match);if(match)matches.add(n.id);}
for(const p of paths)p.element.classList.toggle('muted',!matches.has(p.edge.source)||!matches.has(p.edge.target));});
window.addEventListener('resize',fit);fit();
</script></body></html>"""
