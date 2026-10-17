const fs=require('fs'),vm=require('vm');
const html=fs.readFileSync(require('path').join(__dirname,'../docs/index.html'),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const context=new Proxy({}, {get:(o,p)=>o[p]||((...args)=>{if(args.some(a=>typeof a==='number'&&!Number.isFinite(a)))throw Error('Invalid canvas number');})});
const els={};const sandbox={console,setInterval:()=>0,document:{getElementById(id){return els[id]||(els[id]={value:id==='horizon'?'1':'0',innerHTML:'',textContent:'',getContext:()=>context})}}};
vm.createContext(sandbox);vm.runInContext(script,sandbox);
vm.runInContext(`for(let i=0;i<clips.length;i++){ $('clip').value=String(i);for(const name of Object.keys(clips[i].predictions)){$('model').value=name;for(let k=0;k<clips[i].positions.length;k++){frame=k;draw();}}}for(let h of [1,2]){$('horizon').value=String(h);table();if(($('metrics').innerHTML.match(/<tr>/g)||[]).length!==5)throw Error('Bad table');}`,sandbox);
if(html.includes('NaN'))throw Error('Non-JSON missing value');
console.log('All six clips, five models, animation frames, and both result filters execute with finite canvas coordinates.');
