/** Resolve standard-font aliases omitted from this Windows artifact-tool bundle.
 * Uses the same runtime's packaged Adobe metrics; never changes managed files.
 */
import path from 'node:path';
import fs from 'node:fs/promises';
import {createRequire,registerHooks} from 'node:module';
import {pathToFileURL} from 'node:url';
const runtime=process.env.CODEX_ARTIFACT_RUNTIME || 'C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime';
const require=createRequire(pathToFileURL(path.join(runtime,'dependencies/node/node_modules/@pdf-lib/standard-fonts/package.json')));
const {Font}=require('./lib/Font.js');
const names=['Courier','Courier-Bold','Courier-BoldOblique','Courier-Oblique','Helvetica','Helvetica-Bold','Helvetica-BoldOblique','Helvetica-Oblique','Symbol','Times-Bold','Times-BoldItalic','Times-Italic','Times-Roman','ZapfDingbats'];
registerHooks({
  resolve(specifier,context,next){
    if(specifier.startsWith('#standard-fonts/'))return {url:'artifact-standard-font:'+specifier.slice(16),shortCircuit:true};
    return next(specifier,context);
  },
  load(url,context,next){
    if(!url.startsWith('artifact-standard-font:'))return next(url,context);
    const alias=url.slice(23),name=names.find(n=>n.replaceAll('-','')===alias);
    if(!name)throw new Error('Unknown packaged standard font '+alias);
    const f=Font.load(name),glyphs=f.CharMetrics.map(v=>v.N),indices=new Map(glyphs.map((g,i)=>[g,i]));
    const groups=new Map();
    for(const [a,b,v] of f.KernPairs || []){
      if(!indices.has(a)||!indices.has(b))continue;
      const codes=groups.get(v)||[];codes.push(indices.get(a)*glyphs.length+indices.get(b));groups.set(v,codes);
    }
    const kernPairs=[];
    for(const [v,codes] of groups){codes.sort((a,b)=>a-b);let last=0;kernPairs.push(v,codes.map(c=>{const d=c-last;last=c;return d;}));}
    const data={name,bbox:f.FontBBox,ascender:f.Ascender||0,descender:f.Descender||0,xHeight:f.XHeight||0,capHeight:f.CapHeight||0,glyphNames:glyphs.join(' '),glyphWidths:f.CharMetrics.map(v=>v.WX),kernPairs};
    return {format:'commonjs',source:'module.exports='+JSON.stringify(data),shortCircuit:true};
  }
});
const {registerPdfFontFace}=await import(pathToFileURL(path.join(runtime,'dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
for(const [family,stem] of [['Arial','arial'],['Consolas','consola'],['Calibri','calibri'],['Times New Roman','times']]){
  const variants=stem==='consola'?['consola','consolab','consolai','consolaz']:stem==='times'?['times','timesbd','timesi','timesbi']:[stem,stem==='arial'?'arialbd':'calibrib',stem+'i',stem==='arial'?'arialbi':'calibriz'];
  for(let i=0;i<variants.length;i++){
    registerPdfFontFace({family,weight:i%2?700:400,style:i>1?'italic':'normal',source:new Uint8Array(await fs.readFile('C:/Windows/Fonts/'+variants[i]+'.ttf'))});
  }
}
