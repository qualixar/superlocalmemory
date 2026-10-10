import {SignJWT,jwtVerify} from 'jose';
import {AUTH_ISSUER} from './authorization-policy.ts';
export const GITHUB_CALLBACK=AUTH_ISSUER+'/github/callback';
export function githubAuthorizationUrl(clientId:string,state:string,challenge:string):string {
 const url=new URL('https://github.com/login/oauth/authorize');
 for(const [key,value] of Object.entries({client_id:clientId,redirect_uri:GITHUB_CALLBACK,state,code_challenge:challenge,code_challenge_method:'S256'}))url.searchParams.set(key,value);
 return url.href;
}
export async function exchangeGithubCode(clientId:string,clientSecret:string,code:string,verifier:string,fetcher:typeof fetch=fetch):Promise<string>{
 if(!clientId||!clientSecret)throw new Error('identity_client_configuration');
 if(!code||code.length>4096||!/^[-A-Za-z0-9_.~]{43,128}$/.test(verifier))throw new Error('identity_exchange_failed');
 let response:Response;
 try{response=await fetcher('https://github.com/login/oauth/access_token',{method:'POST',headers:{Accept:'application/json','Content-Type':'application/x-www-form-urlencoded','User-Agent':'SuperLocalMemory'},body:new URLSearchParams({client_id:clientId,client_secret:clientSecret,code,redirect_uri:GITHUB_CALLBACK,code_verifier:verifier}),redirect:'manual',signal:AbortSignal.timeout(10000)});}catch{throw new Error('identity_exchange_failed');}
 if(response.status>=300&&response.status<400)throw new Error('identity_exchange_failed');
 const raw=await response.text();if(raw.length>16384)throw new Error('identity_exchange_failed');
 let value:unknown;try{value=JSON.parse(raw);}catch{throw new Error('identity_exchange_failed');}
 if(value&&typeof value==='object'&&'error' in value){if(value.error==='incorrect_client_credentials'||value.error==='redirect_uri_mismatch')throw new Error('identity_client_configuration');if(value.error==='bad_verification_code')throw new Error('identity_code_rejected');if(value.error==='unverified_user_email')throw new Error('identity_email_unverified');}
 if(!response.ok)throw new Error('identity_provider_http_'+response.status);
 if(!value||typeof value!=='object'||!('access_token' in value)||typeof value.access_token!=='string'||!value.access_token||value.access_token.length>4096||/[\r\n]/.test(value.access_token))throw new Error('identity_exchange_failed');
 return value.access_token;
}
export function escapeHtml(value:string):string {return value.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));}
const PAGE_STYLE = `
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0b1020;color:#edf0f7}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:32px 20px;background:radial-gradient(ellipse at top,#26203e 0,#0b1020 65%)}main{width:100%;max-width:560px;background:#121827;border:1px solid #30364a;border-radius:20px;padding:32px;box-shadow:0 24px 80px #0005}.brand{display:flex;align-items:center;gap:12px;margin-bottom:28px}.mark{display:grid;place-items:center;width:42px;height:42px;border-radius:12px;background:#9a6bff;color:#fff;font-weight:800;font-size:13px}.brand strong{font-size:17px}.byline{display:block;color:#a7b0c4;font-size:12px;margin-top:3px}h1{font-size:28px;line-height:1.2;letter-spacing:-.6px;margin:0 0 12px}p{color:#b9c2d6;line-height:1.6;margin:12px 0}.identity{background:#0e1422;border:1px solid #30364a;border-radius:12px;padding:14px 16px;margin:22px 0}dl{margin:0;display:grid;grid-template-columns:110px 1fr;gap:10px;font-size:14px}dt{color:#a7b0c4}dd{margin:0;overflow-wrap:anywhere}.note{font-size:13px}.permissions{padding-left:20px;color:#d9dfed;line-height:1.6}.options{display:grid;gap:12px;margin:18px 0}label{display:flex;gap:10px;align-items:flex-start;line-height:1.5}select{font:inherit;padding:10px;border:1px solid #48516a;border-radius:8px;background:#0e1422;color:#edf0f7;max-width:100%}input[type=checkbox]{accent-color:#9a6bff;width:18px;height:18px;margin-top:3px;flex-shrink:0}.actions{display:flex;gap:10px;margin-top:24px}button,.button{font:inherit;font-weight:650;padding:12px 18px;border-radius:10px;border:1px solid #48516a;cursor:pointer;color:#edf0f7;background:#1a2235;text-decoration:none}button[name=decision][value=allow],.primary{background:#9260ef;border-color:#a87bf9;flex:1}button:focus-visible,a:focus-visible{outline:3px solid #c7a7ff;outline-offset:3px}details{margin-top:22px;color:#a7b0c4;font-size:12px;overflow-wrap:anywhere}summary{cursor:pointer}code{color:#c4aaff}.footer{margin-top:22px;font-size:12px;color:#a7b0c4;border-top:1px solid #30364a;padding-top:16px}@media(max-width:480px){body{padding:16px}main{padding:24px 20px}h1{font-size:25px}dl{grid-template-columns:90px 1fr}.actions{flex-wrap:wrap}}
`;
export function renderAuthPage(title:string,content:string):string {
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="referrer" content="strict-origin"><title>'+escapeHtml(title)+' · SuperLocalMemory</title><style>'+PAGE_STYLE+'</style></head><body><main><div class="brand"><span class="mark" aria-hidden="true">SLM</span><div><strong>SuperLocalMemory</strong><span class="byline">by Qualixar</span></div></div>'+content+'<p class="footer">Optional web connections. Your local SLM continues to work without sign-in.</p></main></body></html>';
}
/** What is always granted vs. only granted by an explicit choice above; the page
 * has no script, so it must never present requested optional scopes as granted. */
function technicalScopes(scope:readonly string[]):string {
 const granted=scope.filter(s=>s==='slm:read'||s==='slm:connect');const optional=scope.filter(s=>['slm:write','slm:session','slm:mesh','slm:media'].includes(s));
 return '<p>Granted: '+granted.map(escapeHtml).join(', ')+'</p>'+(optional.length?'<p>Only if selected above: '+optional.map(escapeHtml).join(', ')+'</p>':'');
}
export function renderConsentPage(handle:string,description:{clientName:string;redirectHostname:string;redirectIsLoopback:boolean;scope:string[];clientDomain?:string;profileId?:string}):string {
 const native=description.scope.length===1&&description.scope[0]==='slm:connect';
 const permissions=native?'<li>Link this computer and local profile to your SLM account.</li>':'<li>Allow this application to read approved memories from your selected profile.</li>';
 const options=(description.scope.includes('slm:write')?'<label><input type="checkbox" name="write" value="yes"> Allow saving memories</label>':'')+(description.scope.includes('slm:session')?'<label><input type="checkbox" name="session" value="yes"> Allow session tools</label>':'')+(description.scope.includes('slm:mesh')?'<label><input type="checkbox" name="mesh" value="yes"> Allow talking to your other bots</label>':'')+(description.scope.includes('slm:media')?'<label><input type="checkbox" name="media" value="yes"> Allow images and documents</label>':'');
 return renderAuthPage(native?'Link this computer':'Connect your AI','<h1>'+(native?'Link this computer':'Connect your AI')+'</h1><p>Sign in with GitHub to '+(native?'connect your local SLM profile. Your database stays on this computer.':'choose your SLM connection and approve access for this application.')+'</p><div class="identity"><dl><dt>Application</dt><dd>'+escapeHtml(description.clientName)+'</dd>'+(description.profileId?'<dt>Local profile</dt><dd>'+escapeHtml(description.profileId)+'</dd>':'')+'<dt>Returns to</dt><dd>'+escapeHtml(description.redirectHostname)+(description.redirectIsLoopback?' (this computer)':'')+'</dd></dl></div><p class="note">'+(description.clientDomain?'Client domain: '+escapeHtml(description.clientDomain):'The application name is supplied by the client.')+'</p><h2>What you are approving</h2><ul class="permissions">'+permissions+'</ul><p class="note">'+(native?'Only continue if you started this connection in your SLM dashboard. AI memory access requires its own approval.':'Approved memory results will be shared with this application. Saving, session, bot and image tools remain off unless you select them below.')+'</p><form method="post" action="/consent"><input type="hidden" name="handle" value="'+escapeHtml(handle)+'"><div class="options">'+options+'</div><div class="actions"><button name="decision" value="allow">Sign in with GitHub</button><button name="decision" value="deny">Cancel</button></div></form><details><summary>Technical permissions</summary>'+technicalScopes(description.scope)+'</details>');
}
// Shown when a consent or profile form is submitted again after it already
// finished (Back, refresh, a second tab). The grant is already in place.
export function renderSignInComplete():string {
 return renderAuthPage('Sign-in complete','<h1>Sign-in complete</h1><p>This sign-in already finished. You can close this page and go back to your app.</p><div class="identity"><strong>To check the connection</strong><p>Open <strong>Connected apps</strong> in your SLM dashboard: the app is listed there with the permissions you approved. If it is missing, add the app again from your app.</p></div>');
}
export function renderAuthFailure(error:string,dashboard?:string):string {
 const message=error==='identity_client_configuration'?'SLM’s GitHub sign-in configuration needs attention. Your local SLM remains available. Contact support with the code below; repeated sign-in attempts will not fix this.':['consent_unavailable','connection_unavailable','sign_in_session_unavailable'].includes(error)?'This sign-in session is no longer valid. Refreshing it cannot restart authorization.':'We could not complete sign-in. Return to SLM to retry; if this continues, share the support code below.';
 return renderAuthPage('Continue sign-in from SLM','<h1>Continue sign-in from SLM</h1><p>'+message+'</p><div class="identity"><strong>Return to your SLM dashboard</strong><p>Select <strong>Restart sign-in</strong> on your connection. SLM will replace the old attempt using the same approved permissions.</p></div>'+(dashboard&&/^http:\/\/127\.0\.0\.1:\d{1,5}\/#mcp-pane$/.test(dashboard)?'<p><a class="button primary" href="'+escapeHtml(dashboard)+'">Return to SLM</a></p>':'')+'<p>If GitHub asks you to log in, complete login on github.com. Do not enter your GitHub password on this page.</p><details><summary>Support information</summary><p>Code: <code>'+escapeHtml(error)+'</code></p></details>');
}

// A routing receipt only: never accepted as an OAuth token or access proof.
const RETURN_COOKIE='__Host-slm-dashboard-return';
async function returnKey(secret:string):Promise<Uint8Array|null>{
 if(!/^[a-f0-9]{64}$/.test(secret))return null;
 return new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode('slm-dashboard-return-v1:'+secret)));
}
export async function dashboardReturnCookie(uri:string,secret:string):Promise<string|null>{
 const key=await returnKey(secret);if(!key)return null;
 let target:URL;try{target=new URL(uri);}catch{return null;}
 const port=Number(target.port||80);
 if(target.protocol!=='http:'||target.hostname!=='127.0.0.1'||target.pathname!=='/api/v3/connections/callback'||target.username||target.password||target.search||target.hash||!Number.isInteger(port)||port<1||port>65535)return null;
 const token=await new SignJWT({port}).setProtectedHeader({alg:'HS256',typ:'slm-ui-return'}).setIssuer(AUTH_ISSUER).setAudience('slm-dashboard-return').setIssuedAt().setExpirationTime('1h').sign(key);
 return RETURN_COOKIE+'='+token+'; Path=/; Secure; HttpOnly; SameSite=Lax; Max-Age=3600';
}
export async function dashboardReturnUrl(request:Request,secret:string):Promise<string|undefined>{
 const key=await returnKey(secret);if(!key)return undefined;
 const token=request.headers.get('Cookie')?.split(';').map(x=>x.trim()).find(x=>x.startsWith(RETURN_COOKIE+'='))?.slice(RETURN_COOKIE.length+1);
 if(!token||token.length>2048)return undefined;
 try{const {payload,protectedHeader}=await jwtVerify(token,key,{issuer:AUTH_ISSUER,audience:'slm-dashboard-return',algorithms:['HS256']});const port=payload.port;if(protectedHeader.typ!=='slm-ui-return'||typeof port!=='number'||!Number.isInteger(port)||port<1||port>65535)return undefined;return 'http://127.0.0.1:'+port+'/#mcp-pane';}catch{return undefined;}
}
