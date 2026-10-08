// Hosted mode uses typed server transactions and the same graph rendered by offline preview.
const host=JSON.parse(document.getElementById('host').textContent);
let digest=host.digest, currentJob=null, jobKey=null, busy=false, editFailed=false, mutation=Promise.resolve(), chosenEdge=null, wireStart=null;
const specs=new Map(host.catalogue.map(s=>[s.operator,s]));
const byId=id=>document.getElementById(id);
function dom(tag,parent,text,attrs={}){const element=document.createElement(tag);element.textContent=text||'';for(const [key,value] of Object.entries(attrs))element.setAttribute(key,value);parent.appendChild(element);return element;}
const controls=dom('div',document.querySelector('.toolbar'),'',{class:'host-controls'});
dom('button',controls,'Execute',{id:'execute'});dom('button',controls,'Cancel job',{id:'cancelJob',disabled:''});
dom('label',controls,'Seed',{for:'seed'});dom('input',controls,'',{id:'seed',type:'number',value:'42',min:'0',step:'1'});
dom('button',controls,'Remove node',{id:'removeNode',disabled:''});dom('button',controls,'Remove connection',{id:'removeEdge',disabled:''});
dom('p',controls,host.allow_external?'Provider generation enabled for this runtime.':'Local execution enabled. Start the host with --allow-external to enable provider generation.',{class:'host-note'});
dom('p',controls,'Drag an output circle to an input circle to connect. Or select both ports with Enter. Layout/config edits are saved automatically.',{class:'host-note'});
dom('p',controls,'',{id:'hostError',role:'alert'});
document.querySelector('aside .hint').textContent='Saved canvas · select a node to inspect or configure';
const paletteBox=dom('details',document.querySelector('aside'),'',{id:'palette'});dom('summary',paletteBox,'Add an operator');
dom('label',paletteBox,'Operator',{for:'operator'});const operatorSelect=dom('select',paletteBox,'',{id:'operator'});
for(const spec of host.catalogue)dom('option',operatorSelect,spec.operator,{value:spec.operator});
dom('label',paletteBox,'Node ID',{for:'newId'});dom('input',paletteBox,'',{id:'newId',value:'node-'+doc.nodes.length});
dom('label',paletteBox,'Initial configuration JSON',{for:'newConfig'});dom('textarea',paletteBox,'',{id:'newConfig'});
dom('button',paletteBox,'Add node',{id:'addNode'});
const jobPanel=dom('section',document.querySelector('aside'),'',{id:'jobPanel','aria-live':'polite'});
dom('h2',jobPanel,'Execution');dom('p',jobPanel,'No job yet',{id:'jobStatus'});const jobResults=dom('div',jobPanel,'',{id:'jobResults'});

async function api(path,payload){const response=await fetch(path,{method:payload===undefined?'GET':'POST',headers:{'X-Vidliner-Token':host.token,'Content-Type':'application/json'},body:payload===undefined?undefined:JSON.stringify(payload)});const value=await response.json();if(!response.ok)throw new Error(value.error||'Host request failed');return value;}
function error(value){byId('hostError').textContent=value?.message||String(value||'');}
function adopt(value){Object.assign(doc,value.document);digest=value.digest;const id=selected?.id;redrawGraph();if(id&&index.has(id))select(index.get(id));else{selected=null;byId('configEditor').hidden=true;}byId('removeNode').disabled=!selected;}
function edit(makeEdits){mutation=mutation.then(async()=>{const value=await api('/api/edits',{expected_digest:digest,edits:makeEdits()});adopt(value);editFailed=false;byId('status').textContent='Saved · validated graph';error('');}).catch(async failure=>{editFailed=true;error(failure);try{adopt(await api('/api/workflow'));}catch(next){error(next);}});return mutation;}
function portY(n,port,direction){const names=Object.keys(direction==='output'?n.outputs:specs.get(n.operator).inputs);return 112+names.indexOf(port)*22;}
edgePath=edge=>{const a=index.get(edge.source),b=index.get(edge.target);const ay=a.position.y+portY(a,edge.source_port,'output'),by=b.position.y+portY(b,edge.target_port,'input');return `M ${a.position.x+256} ${ay} C ${a.position.x+310} ${ay}, ${b.position.x-54} ${by}, ${b.position.x} ${by}`;};
function redrawGraph(){scene.replaceChildren();index.clear();groups.clear();paths.length=0;chosenEdge=null;byId('removeEdge').disabled=true;
 for(const n of doc.nodes)index.set(n.id,n);
 for(const edge of doc.edges){const path=el('path',{d:edgePath(edge),class:'edge',tabindex:0,role:'button','aria-label':`${edge.source}:${edge.source_port} to ${edge.target}:${edge.target_port}`},scene);paths.push({element:path,edge});const choose=()=>{chosenEdge=edge;for(const p of paths)p.element.classList.toggle('chosen',p.edge===edge);byId('removeEdge').disabled=false;};path.onpointerdown=e=>e.stopPropagation();path.onclick=choose;path.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}};}
 for(const n of doc.nodes){const spec=specs.get(n.operator),height=124+Math.max(Object.keys(spec.inputs).length,Object.keys(n.outputs).length)*22;
  const g=el('g',{transform:`translate(${n.position.x},${n.position.y})`,class:'node',tabindex:0,role:'button','aria-label':n.operator+' '+n.id},scene);groups.set(n.id,g);
  el('rect',{width:256,height,rx:8,fill:'#152338',stroke:palette[n.stage]||'#6584b0'},g);
  el('text',{x:16,y:25,fill:palette[n.stage]||'#98abc8','font-size':11},g,n.stage.toUpperCase());
  el('text',{x:16,y:50,fill:'#e3ebf7','font-size':14},g,n.operator);el('text',{x:16,y:75,fill:'#98abc8','font-size':11},g,n.id);
  el('text',{x:150,y:25,class:'run-state','data-state':n.id},g,'');
  g.onclick=e=>{if(e.target.classList.contains('port'))return;select(n);byId('removeNode').disabled=false;};
  g.onkeydown=e=>{if(e.target!==g)return;if(e.key==='Enter'||e.key===' '){e.preventDefault();select(n);byId('removeNode').disabled=false;}};
  for(const direction of ['input','output']){const entries=Object.entries(direction==='output'?n.outputs:spec.inputs);
   for(const [port,binding] of entries){const y=portY(n,port,direction),x=direction==='output'?256:0,type=direction==='output'?binding:binding.type;
    el('text',{x:direction==='output'?242:14,y:y+4,fill:'#b9cbe4','font-size':10,'text-anchor':direction==='output'?'end':'start'},g,port);
    const circle=el('circle',{cx:x,cy:y,r:6,class:'port',tabindex:0,role:'button','aria-label':`${direction} ${n.id}:${port} (${type})`,'data-direction':direction,'data-node':n.id,'data-port':port,'data-type':type},g);
    circle.onpointerdown=e=>{e.stopPropagation();if(direction==='output'){wireStart={node:n.id,port,type};svg.setPointerCapture(e.pointerId);}};
    circle.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();if(direction==='output'){wireStart={node:n.id,port,type};circle.classList.add('chosen');}else if(wireStart)connect(n.id,port,type);}};
   }
  }
  let moving=null;g.onpointerdown=e=>{if(!document.body.classList.contains('editing')||e.button!==0||e.target.classList.contains('port'))return;e.stopPropagation();g.setPointerCapture(e.pointerId);moving={x:e.clientX,y:e.clientY,px:n.position.x,py:n.position.y};};
  g.onpointermove=e=>{if(!moving)return;n.position.x=moving.px+(e.clientX-moving.x)/scale;n.position.y=moving.py+(e.clientY-moving.y)/scale;g.setAttribute('transform',`translate(${n.position.x},${n.position.y})`);for(const p of paths)p.element.setAttribute('d',edgePath(p.edge));};
  g.onpointerup=()=>{if(!moving)return;moving=null;const position={...n.position};edit(()=>[{op:'move_node',node_id:n.id,position}]);};g.onpointercancel=()=>{moving=null;};
 }
 byId('counts').textContent=doc.nodes.length+' nodes · '+doc.edges.length+' connections';draw();renderStates();
}
function connect(target,target_port,type){const start=wireStart;wireStart=null;scene.querySelector('.wire')?.remove();if(!start)return;
 if(start.type!==type&&start.type!=='any'&&type!=='any'){error('Port types do not match');return;}
 edit(()=>{const node=index.get(target),edits=[];for(const edge of doc.edges)if(edge.target===target&&edge.target_port===target_port)edits.push({op:'disconnect',edge});if(node.inputs[target_port])edits.push({op:'unbind_input',node_id:target,port:target_port});edits.push({op:'connect',edge:{source:start.node,source_port:start.port,target,target_port}});return edits;});}
svg.addEventListener('pointermove',e=>{if(!wireStart)return;const n=index.get(wireStart.node),rect=svg.getBoundingClientRect(),x=(e.clientX-rect.left-ox)/scale,y=(e.clientY-rect.top-oy)/scale;let path=scene.querySelector('.wire');if(!path)path=el('path',{class:'wire'},scene);path.setAttribute('d',`M ${n.position.x+256} ${n.position.y+portY(n,wireStart.port,'output')} L ${x} ${y}`);});
svg.addEventListener('pointerup',e=>{if(!wireStart)return;const target=document.elementFromPoint(e.clientX,e.clientY)?.closest('[data-direction="input"]');if(target)connect(target.dataset.node,target.dataset.port,target.dataset.type);else{wireStart=null;scene.querySelector('.wire')?.remove();}});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){wireStart=null;scene.querySelector('.wire')?.remove();scene.querySelectorAll('.chosen').forEach(x=>x.classList.remove('chosen'));}});
byId('applyConfig').onclick=()=>{try{const nodeId=selected.id,config=JSON.parse(byId('config').value);edit(()=>[{op:'configure_node',node_id:nodeId,config}]);}catch(failure){error(failure);}};
byId('resetConfig').onclick=()=>{if(selected)select(selected);editFailed=false;error('');};
byId('removeEdge').onclick=()=>{if(!chosenEdge)return;const edge=chosenEdge;edit(()=>[{op:'disconnect',edge},{op:'bind_input',node_id:edge.target,port:edge.target_port,binding:{kind:'artifact',role:edge.target+'.'+edge.target_port}}]);};
byId('removeNode').onclick=()=>{if(!selected)return;const nodeId=selected.id;edit(()=>[...doc.edges.filter(e=>e.source===nodeId&&e.target!==nodeId).map(e=>({op:'bind_input',node_id:e.target,port:e.target_port,binding:{kind:'artifact',role:e.target+'.'+e.target_port}})),{op:'remove_node',node_id:nodeId}]);};
function paletteDefault(){const name=operatorSelect.value;byId('newConfig').value=JSON.stringify(name==='video.submit'?{request:{prompt:'A quiet mountain lake at sunrise, gentle camera movement',parameters:{duration:5,ratio:'1280:720'}}}:name==='video.plan_variants'?{spec:{id:'canvas',seed:42,operators:[{name:'format.reframe',axis:'format',domains:{aspect:['1:1']}}]}}:{},null,2);}
operatorSelect.value='video.submit';operatorSelect.onchange=paletteDefault;paletteDefault();
byId('addNode').onclick=async()=>{try{const node=await api('/api/nodes',{operator:operatorSelect.value,id:byId('newId').value,config:JSON.parse(byId('newConfig').value)});node.position={x:doc.nodes.length*320,y:160};await edit(()=>[{op:'add_node',node}]);byId('newId').value='node-'+doc.nodes.length;fit();}catch(failure){error(failure);}};
function renderStates(){for(const text of scene.querySelectorAll('[data-state]'))text.textContent=currentJob?.digest===digest?currentJob.nodes[text.dataset.state]?.status||'':'';}
function showJob(job){currentJob=job;const active=['running','polling','cancelling'].includes(job.status);byId('execute').disabled=active||busy;byId('cancelJob').disabled=!active&&!Object.values(job.tasks).some(t=>!['succeeded','failed','cancelled'].includes(t.status));
 byId('jobStatus').textContent=job.job_id+' · '+job.status+(job.error?' · '+job.error:'');renderStates();
 const fingerprint=JSON.stringify({status:job.status,tasks:job.tasks,nodes:Object.fromEntries(Object.entries(job.nodes).map(([id,n])=>[id,n.status]))});if(jobResults.dataset.fingerprint===fingerprint)return;jobResults.dataset.fingerprint=fingerprint;jobResults.replaceChildren();
 for(const [id,node] of Object.entries(job.nodes)){dom('p',jobResults,id+' · '+node.status+(node.error_message?' · '+node.error_message:''));}
 const seen=new Set();for(const [id,task] of Object.entries(job.tasks)){if(seen.has(task.task_id))continue;seen.add(task.task_id);dom('p',jobResults,id+' · '+task.status+' · '+task.model_id);
  for(const [label,method] of [['Refresh task','status'],['Cancel task','cancel']]){const button=dom('button',jobResults,label);button.onclick=async()=>{button.disabled=true;try{showJob(await api(`/api/jobs/${job.job_id}/tasks/${encodeURIComponent(id)}/${method}`,{}));}catch(failure){error(failure);}finally{button.disabled=false;}};}
  for(const url of task.output_urls){const video=dom('video',jobResults,'',{controls:'',preload:'metadata','aria-label':'Generated video '+id});video.src=url;video.referrerPolicy='no-referrer';}if(task.output_urls.length)dom('p',jobResults,'Generated media · training verification pending',{class:'hint'});
 }
}
byId('execute').onclick=async()=>{busy=true;byId('execute').disabled=true;try{await mutation;if(editFailed)throw new Error('Fix the rejected edit or Reset configuration before executing the saved graph');jobKey=jobKey||crypto.randomUUID();const job=await api('/api/execute',{job_id:jobKey,expected_digest:digest,seed:Number(byId('seed').value)});jobKey=null;showJob(job);error('');}catch(failure){error(failure);}finally{busy=false;byId('execute').disabled=['running','polling','cancelling'].includes(currentJob?.status);}};
byId('cancelJob').onclick=async()=>{if(!currentJob)return;try{showJob(await api('/api/jobs/'+currentJob.job_id+'/cancel',{}));}catch(failure){error(failure);}};
setInterval(async()=>{if(!currentJob||!['running','polling','cancelling'].includes(currentJob.status))return;try{showJob(await api('/api/jobs/'+currentJob.job_id));}catch(failure){error(failure);}},2000);
redrawGraph();fit();api('/api/jobs').then(jobs=>{if(jobs.length)showJob(jobs[0]);}).catch(error);
