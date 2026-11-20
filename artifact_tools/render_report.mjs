/** Render the actual DOCX with the bundled document engine when LibreOffice is absent. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import './artifact_runtime_compat.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.CODEX_ARTIFACT_RUNTIME || 'C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime';
const {DocumentFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const input=process.argv[2]?path.resolve(process.argv[2]):path.join(root,'deliverables/CS406_Report.docx');
const output=process.argv[3]?path.resolve(process.argv[3]):path.join(root,'deliverables/CS406_Report.pdf');
const document=await DocumentFile.importDocx(await FileBlob.load(input));
// The importer preserves Word page-number properties. Bridge them into the
// renderer's pageSetup while avoiding inherited restarts at page-break fragments.
let previousFormat;
for(const section of document.sections.items){
  if(!section.pageNumberFormat)continue;
  section.pageSetup={...section.pageSetup,pageNumbers:{
    start:section.pageNumberFormat===previousFormat?undefined:section.pageNumberStart,
    formatType:section.pageNumberFormat
  }};
  previousFormat=section.pageNumberFormat;
}
const pdf=await document.export({format:'pdf'});
await fs.writeFile(output,new Uint8Array(await pdf.arrayBuffer()));
const layout=await document.export({format:'layout'});
await fs.writeFile(path.join(path.dirname(output),'report-layout.json'),await layout.text());
console.log('Rendered actual DOCX to '+output);
