/** The only pages the upload link serves: one small picker, and a plain message. Static text, no third parties, strict CSP. */

const IMAGE_ACCEPT = "image/png,image/jpeg,image/gif,image/webp";
const DOC_ACCEPT = "application/pdf";
const DEFAULT_MAX = 25 * 1024 * 1024;
const BASE_HEADERS = {
  "Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff",
  "X-Robots-Tag": "noindex, nofollow", "Cross-Origin-Opener-Policy": "same-origin",
} as const;

function nonce(): string {
  let text = "";
  for (const byte of crypto.getRandomValues(new Uint8Array(16))) text += String.fromCharCode(byte);
  return btoa(text).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
const STYLE = `:root{color-scheme:light dark}body{margin:0;font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;background:Canvas;color:CanvasText}
main{max-width:30rem;margin:12vh auto 0;padding:0 1.25rem}h1{font-size:1.35rem;margin:0 0 .5rem}p{margin:.5rem 0}
input[type=file]{display:block;width:100%;margin:1rem 0;padding:.75rem;border:1px dashed GrayText;border-radius:.5rem;box-sizing:border-box}
button{font:inherit;padding:.6rem 1.4rem;border:0;border-radius:.5rem;background:#3b4cca;color:#fff;cursor:pointer}button:disabled{opacity:.45;cursor:default}
.ok{color:#1a6b3c}.bad{color:#b3261e}@media(prefers-color-scheme:dark){.ok{color:#7fd6a0}.bad{color:#ff8a80}}[hidden]{display:none}`;

function script(maxMb: number, maxBytes: number): string {
  return `(function(){var MAX=${maxBytes},MB=${maxMb};var file=document.getElementById('file'),save=document.getElementById('save'),out=document.getElementById('status');
function say(text,bad){out.textContent=text;out.className=bad?'bad':'ok';}
file.addEventListener('change',function(){var f=file.files[0];save.disabled=!f;if(f&&f.size>MAX){save.disabled=true;say('That file is too large. The limit is '+MB+' MB.',true);}else{say('',false);}});
save.addEventListener('click',async function(){var f=file.files[0];if(!f)return;save.disabled=true;file.disabled=true;say('Saving. Keep this page open until it says Saved.',false);
try{var r=await fetch(location.pathname,{method:'POST',body:f,headers:{'Content-Type':'application/octet-stream'},credentials:'omit',referrerPolicy:'no-referrer',cache:'no-store'});var j=await r.json();
if(j&&j.ok){say(String(j.message||(j.done?'Saved to your memory.':'Still being saved on your computer. You can close this page.')),false);file.hidden=true;save.hidden=true;}
else{say(String((j&&j.message)||'The file could not be saved.'),true);file.disabled=false;save.disabled=false;}}
catch(e){say('The upload was interrupted. Check your connection and press Save to try again.',true);file.disabled=false;save.disabled=false;}});})();`;
}

function html(status: number, body: string, csp: string, token: string, extraStyle = ""): Response {
  const page = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">` +
    `<meta name="referrer" content="no-referrer"><meta name="robots" content="noindex"><title>SuperLocalMemory</title><style nonce="${token}">${STYLE}${extraStyle}</style></head><body><main>${body}</main></body></html>`;
  return new Response(page, { status, headers: { ...BASE_HEADERS, "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": csp } });
}

/** The picker. `kind` and `maxBytes` come from the laptop, but only the known kinds and a sane size reach the markup. */
export function uploadPage(kind: string, maxBytes: number): Response {
  const document = kind === "document";
  const bytes = Number.isSafeInteger(maxBytes) && maxBytes > 0 ? maxBytes : DEFAULT_MAX;
  const mb = Math.max(1, Math.floor(bytes / (1024 * 1024)));
  const token = nonce();
  const noun = document ? "PDF" : "picture";
  const hint = document ? `Pick a PDF, up to ${mb} MB.` : `Pick a PNG, JPEG, GIF or WebP picture, up to ${mb} MB.`;
  const csp = `default-src 'none'; script-src 'nonce-${token}'; style-src 'nonce-${token}'; connect-src 'self'; form-action 'none'; base-uri 'none'; frame-ancestors 'none'`;
  const body = `<h1>Add a ${noun} to your memory</h1><p>${hint}</p><p>This link works once, expires in 10 minutes, and is only for you. Do not share it with anyone.</p>` +
    `<input id="file" type="file" accept="${document ? DOC_ACCEPT : IMAGE_ACCEPT}"><button id="save" type="button" disabled>Save</button><p id="status" role="status"></p>` +
    `<script nonce="${token}">${script(mb, bytes)}</script>`;
  return html(200, body, csp, token);
}

/** A plain message with no script at all. The text is escaped. */
export function messagePage(status: number, message: string): Response {
  const token = nonce();
  const csp = `default-src 'none'; style-src 'nonce-${token}'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`;
  return html(status, `<h1>SuperLocalMemory</h1><p>${escapeHtml(message)}</p>`, csp, token);
}

/** The JSON answer to the page's upload. */
export function jsonReply(status: number, body: Record<string, unknown>): Response {
  return new Response(JSON.stringify(body), { status, headers: { ...BASE_HEADERS, "Content-Type": "application/json" } });
}
