// Hosted mode uses typed server transactions and the same graph rendered by offline preview.
const host=JSON.parse(document.getElementById('host').textContent);
const localeRuntime=window.VidLinerI18n;
let digest=host.digest, currentJob=null, jobKey=null, busy=false, editFailed=false, mutation=Promise.resolve(), chosenEdge=null, wireStart=null;
const specs=new Map(host.catalogue.map(s=>[s.operator,s]));
const byId=id=>document.getElementById(id);
const t=(key,values)=>localeRuntime.t(key,values);
const statusLabel=status=>t('status.'+status,{default:status});
function dom(tag,parent,text,attrs={}){const element=document.createElement(tag);element.textContent=text||'';for(const [key,value] of Object.entries(attrs))element.setAttribute(key,value);parent.appendChild(element);return element;}
const controls=dom('div',document.querySelector('.toolbar'),'',{class:'host-controls'});
byId('status').removeAttribute('data-i18n');
dom('button',controls,t('toolbar.execute'),{id:'execute','data-i18n':'toolbar.execute'});dom('button',controls,t('toolbar.cancelJob'),{id:'cancelJob',disabled:'','data-i18n':'toolbar.cancelJob'});
dom('label',controls,t('toolbar.seed'),{for:'seed','data-i18n':'toolbar.seed'});dom('input',controls,'',{id:'seed',type:'number',value:'42',min:'0',step:'1'});
dom('button',controls,t('toolbar.removeNode'),{id:'removeNode',disabled:'','data-i18n':'toolbar.removeNode'});dom('button',controls,t('toolbar.removeConnection'),{id:'removeEdge',disabled:'','data-i18n':'toolbar.removeConnection'});
dom('p',controls,host.allow_external?t('toolbar.providerEnabled'):t('toolbar.localExecution'),{class:'host-note','data-i18n':host.allow_external?'toolbar.providerEnabled':'toolbar.localExecution'});
dom('p',controls,t('toolbar.connectHelp'),{class:'host-note','data-i18n':'toolbar.connectHelp'});
dom('p',controls,'',{id:'hostError',role:'alert'});
document.querySelector('aside .hint').textContent=t('aside.savedCanvas');document.querySelector('aside .hint').dataset.i18n='aside.savedCanvas';
const paletteBox=dom('details',document.querySelector('aside'),'',{id:'palette'});dom('summary',paletteBox,t('aside.addOperator'),{'data-i18n':'aside.addOperator'});
dom('label',paletteBox,t('aside.operator'),{for:'operator','data-i18n':'aside.operator'});const operatorSelect=dom('select',paletteBox,'',{id:'operator'});
for(const spec of host.catalogue)dom('option',operatorSelect,spec.operator,{value:spec.operator});
dom('label',paletteBox,t('aside.nodeId'),{for:'newId','data-i18n':'aside.nodeId'});dom('input',paletteBox,'',{id:'newId',value:'node-'+doc.nodes.length});
dom('label',paletteBox,t('aside.initialConfiguration'),{for:'newConfig','data-i18n':'aside.initialConfiguration'});dom('textarea',paletteBox,'',{id:'newConfig'});
dom('button',paletteBox,t('aside.addNode'),{id:'addNode','data-i18n':'aside.addNode'});
const jobPanel=dom('section',document.querySelector('aside'),'',{id:'jobPanel','aria-live':'polite'});
dom('h2',jobPanel,t('aside.executionTrace'),{ 'data-i18n':'aside.executionTrace'});dom('p',jobPanel,t('aside.noRun'),{id:'jobStatus',class:'job-status','data-i18n':'aside.noRun'});const jobResults=dom('div',jobPanel,'',{id:'jobResults'});

async function api(path,payload){const response=await fetch(path,{method:payload===undefined?'GET':'POST',headers:{'X-Vidliner-Token':host.token,'Content-Type':'application/json'},body:payload===undefined?undefined:JSON.stringify(payload)});const value=await response.json();if(!response.ok)throw new Error(value.error||t('error.hostRequest'));return value;}
function error(value){const message=value?.message||String(value||'');const translated=message==='Port types do not match'?t('status.portMismatch'):message==='Fix the rejected edit or Reset configuration before executing the saved graph'?t('error.executionRejected'):message;byId('hostError').textContent=translated||'';}
function adopt(value){Object.assign(doc,value.document);digest=value.digest;const id=selected?.id;redrawGraph();if(id&&index.has(id))select(index.get(id));else{selected=null;byId('configEditor').hidden=true;}byId('removeNode').disabled=!selected;}
function edit(makeEdits){mutation=mutation.then(async()=>{const value=await api('/api/edits',{expected_digest:digest,edits:makeEdits()});adopt(value);editFailed=false;byId('status').textContent=t('status.saved');error('');}).catch(async failure=>{editFailed=true;error(failure);try{adopt(await api('/api/workflow'));}catch(next){error(next);}});return mutation;}
function portY(n,port,direction){const names=Object.keys(direction==='output'?n.outputs:specs.get(n.operator).inputs);return 112+names.indexOf(port)*22;}
edgePath=edge=>{const a=index.get(edge.source),b=index.get(edge.target);const ay=a.position.y+portY(a,edge.source_port,'output'),by=b.position.y+portY(b,edge.target_port,'input');return `M ${a.position.x+256} ${ay} C ${a.position.x+310} ${ay}, ${b.position.x-54} ${by}, ${b.position.x} ${by}`;};
function redrawGraph(){scene.replaceChildren();index.clear();groups.clear();paths.length=0;chosenEdge=null;byId('removeEdge').disabled=true;
 for(const n of doc.nodes)index.set(n.id,n);
 for(const edge of doc.edges){const path=el('path',{d:edgePath(edge),class:'edge',tabindex:0,role:'button','aria-label':`${edge.source}:${edge.source_port} to ${edge.target}:${edge.target_port}`},scene);paths.push({element:path,edge});const choose=()=>{chosenEdge=edge;for(const p of paths)p.element.classList.toggle('chosen',p.edge===edge);byId('removeEdge').disabled=false;};path.onpointerdown=e=>e.stopPropagation();path.onclick=choose;path.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}};}
 for(const n of doc.nodes){const spec=specs.get(n.operator),height=124+Math.max(Object.keys(spec.inputs).length,Object.keys(n.outputs).length)*22;
  const g=el('g',{transform:`translate(${n.position.x},${n.position.y})`,class:'node',tabindex:0,role:'button','aria-label':n.operator+' '+n.id},scene);groups.set(n.id,g);
  el('rect',{width:256,height,rx:2,fill:'#fffdf7',stroke:palette[n.stage]||'#9aa69a','stroke-width':2},g);
  el('text',{x:16,y:25,fill:palette[n.stage]||'#6e7b70','font-size':11,'data-stage':n.stage},g,t('stage.'+n.stage));
  el('text',{x:16,y:50,fill:'#181b24','font-size':14},g,n.operator);el('text',{x:16,y:75,fill:'#6e7b70','font-size':11},g,n.id);
  el('text',{x:150,y:25,class:'run-state','data-state':n.id},g,'');
  g.onclick=e=>{if(e.target.classList.contains('port'))return;select(n);byId('removeNode').disabled=false;};
  g.onkeydown=e=>{if(e.target!==g)return;if(e.key==='Enter'||e.key===' '){e.preventDefault();select(n);byId('removeNode').disabled=false;}};
  for(const direction of ['input','output']){const entries=Object.entries(direction==='output'?n.outputs:spec.inputs);
   for(const [port,binding] of entries){const y=portY(n,port,direction),x=direction==='output'?256:0,type=direction==='output'?binding:binding.type;
    el('text',{x:direction==='output'?242:14,y:y+4,fill:'#6e7b70','font-size':10,'text-anchor':direction==='output'?'end':'start'},g,port);
    const circle=el('circle',{cx:x,cy:y,r:6,class:'port',tabindex:0,role:'button','aria-label':`${direction} ${n.id}:${port} (${type})`,'data-direction':direction,'data-node':n.id,'data-port':port,'data-type':type},g);
    circle.onpointerdown=e=>{e.stopPropagation();if(direction==='output'){wireStart={node:n.id,port,type};svg.setPointerCapture(e.pointerId);}};
    circle.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();if(direction==='output'){wireStart={node:n.id,port,type};circle.classList.add('chosen');}else if(wireStart)connect(n.id,port,type);}};
   }
  }
  let moving=null;g.onpointerdown=e=>{if(!document.body.classList.contains('editing')||e.button!==0||e.target.classList.contains('port'))return;e.stopPropagation();g.setPointerCapture(e.pointerId);moving={x:e.clientX,y:e.clientY,px:n.position.x,py:n.position.y};};
  g.onpointermove=e=>{if(!moving)return;n.position.x=moving.px+(e.clientX-moving.x)/scale;n.position.y=moving.py+(e.clientY-moving.y)/scale;g.setAttribute('transform',`translate(${n.position.x},${n.position.y})`);for(const p of paths)p.element.setAttribute('d',edgePath(p.edge));};
  g.onpointerup=()=>{if(!moving)return;moving=null;const position={...n.position};edit(()=>[{op:'move_node',node_id:n.id,position}]);};g.onpointercancel=()=>{moving=null;};
 }
 byId('counts').textContent=t('counts',{nodes:new Intl.NumberFormat(localeRuntime.locale()).format(doc.nodes.length),connections:new Intl.NumberFormat(localeRuntime.locale()).format(doc.edges.length)});draw();renderStates();
}
function connect(target,target_port,type){const start=wireStart;wireStart=null;scene.querySelector('.wire')?.remove();if(!start)return;
 if(start.type!==type&&start.type!=='any'&&type!=='any'){error(t('status.portMismatch'));return;}
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
function renderStates(){for(const text of scene.querySelectorAll('[data-state]')){const status=currentJob?.digest===digest?currentJob.nodes[text.dataset.state]?.status||'':'';text.textContent=status?statusLabel(status):'';}}
function showJob(job){currentJob=job;const active=['running','polling','cancelling'].includes(job.status);byId('execute').disabled=active||busy;byId('cancelJob').disabled=!active&&!Object.values(job.tasks).some(t=>!['succeeded','failed','cancelled'].includes(t.status));
 byId('jobStatus').textContent=t('jobStatus',{job:job.job_id,status:statusLabel(job.status),error:job.error?' · '+job.error:''});renderStates();
 const fingerprint=JSON.stringify({status:job.status,tasks:job.tasks,nodes:Object.fromEntries(Object.entries(job.nodes).map(([id,n])=>[id,n.status]))});if(jobResults.dataset.fingerprint===fingerprint)return;jobResults.dataset.fingerprint=fingerprint;jobResults.replaceChildren();
 for(const [id,node] of Object.entries(job.nodes)){dom('p',jobResults,t('nodeStatus',{node:id,status:statusLabel(node.status),error:node.error_message?' · '+node.error_message:''}));}
 const seen=new Set();for(const [id,task] of Object.entries(job.tasks)){if(seen.has(task.task_id))continue;seen.add(task.task_id);dom('p',jobResults,t('taskStatus',{node:id,status:statusLabel(task.status),model:task.model_id}));
  for(const [key,method] of [['refreshTask','status'],['cancelTask','cancel']]){const button=dom('button',jobResults,t(key),{'data-i18n':key});button.onclick=async()=>{button.disabled=true;try{showJob(await api(`/api/jobs/${job.job_id}/tasks/${encodeURIComponent(id)}/${method}`,{}));}catch(failure){error(failure);}finally{button.disabled=false;}};}
  for(const url of task.output_urls){const video=dom('video',jobResults,'',{controls:'',preload:'metadata','aria-label':t('generatedVideo',{node:id})});video.src=url;video.referrerPolicy='no-referrer';}if(task.output_urls.length)dom('p',jobResults,t('aside.generatedMedia'),{class:'hint','data-i18n':'aside.generatedMedia'});
 }
}
byId('execute').onclick=async()=>{busy=true;byId('execute').disabled=true;try{await mutation;if(editFailed)throw new Error('Fix the rejected edit or Reset configuration before executing the saved graph');jobKey=jobKey||crypto.randomUUID();const job=await api('/api/execute',{job_id:jobKey,expected_digest:digest,seed:Number(byId('seed').value)});jobKey=null;showJob(job);error('');}catch(failure){error(failure);}finally{busy=false;byId('execute').disabled=['running','polling','cancelling'].includes(currentJob?.status);}};
byId('cancelJob').onclick=async()=>{if(!currentJob)return;try{showJob(await api('/api/jobs/'+currentJob.job_id+'/cancel',{}));}catch(failure){error(failure);}};
setInterval(async()=>{if(!currentJob||!['running','polling','cancelling'].includes(currentJob.status))return;try{showJob(await api('/api/jobs/'+currentJob.job_id));}catch(failure){error(failure);}},2000);
window.addEventListener('vidliner:locale-change',()=>{byId('counts').textContent=t('counts',{nodes:new Intl.NumberFormat(localeRuntime.locale()).format(doc.nodes.length),connections:new Intl.NumberFormat(localeRuntime.locale()).format(doc.edges.length)});for(const stage of scene.querySelectorAll('[data-stage]'))stage.textContent=t('stage.'+stage.dataset.stage);renderStates();if(currentJob)showJob(currentJob);});
localeRuntime.init();redrawGraph();fit();api('/api/jobs').then(jobs=>{if(jobs.length)showJob(jobs[0]);}).catch(error);
