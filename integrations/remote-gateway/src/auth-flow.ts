import {AUTH_ISSUER} from './authorization-policy.ts';
export const GITHUB_CALLBACK=AUTH_ISSUER+'/github/callback';
export function githubAuthorizationUrl(clientId:string,state:string,challenge:string):string {
 const url=new URL('https://github.com/login/oauth/authorize');
 for(const [key,value] of Object.entries({client_id:clientId,redirect_uri:GITHUB_CALLBACK,state,code_challenge:challenge,code_challenge_method:'S256'}))url.searchParams.set(key,value);
 return url.href;
}
export async function exchangeGithubCode(clientId:string,clientSecret:string,code:string,verifier:string,fetcher:typeof fetch=fetch):Promise<string>{
 if(!clientId||!clientSecret||!code||code.length>4096||!/^[-A-Za-z0-9_.~]{43,128}$/.test(verifier))throw new Error('identity_exchange_failed');
 let response:Response;
 try{response=await fetcher('https://github.com/login/oauth/access_token',{method:'POST',headers:{Accept:'application/json','Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({client_id:clientId,client_secret:clientSecret,code,redirect_uri:GITHUB_CALLBACK,code_verifier:verifier}),redirect:'error',signal:AbortSignal.timeout(10000)});}catch{throw new Error('identity_exchange_failed');}
 if(!response.ok)throw new Error('identity_exchange_failed');
 const raw=await response.text();if(raw.length>16384)throw new Error('identity_exchange_failed');
 let value:unknown;try{value=JSON.parse(raw);}catch{throw new Error('identity_exchange_failed');}
 if(!value||typeof value!=='object'||!('access_token' in value)||typeof value.access_token!=='string'||!value.access_token||value.access_token.length>4096||/[\r\n]/.test(value.access_token))throw new Error('identity_exchange_failed');
 return value.access_token;
}
export function escapeHtml(value:string):string {return value.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));}
export function renderConsentPage(handle:string,description:{clientName:string;redirectHostname:string;redirectIsLoopback:boolean;scope:string[];clientDomain?:string}):string {
 return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Connect SuperLocalMemory</title><main><h1>Connect SuperLocalMemory</h1><p>Application: '+escapeHtml(description.clientName)+'</p>'+(description.clientDomain?'<p>Verified client domain: '+escapeHtml(description.clientDomain)+'</p>':'<p>The application name is supplied by the client.</p>')+'<p>Callback: '+escapeHtml(description.redirectHostname)+'</p>'+(description.redirectIsLoopback?'<p>This sends authorization to an application on your computer. Confirm that you started this connection.</p>':'')+'<p>Requested permissions: '+description.scope.map(escapeHtml).join(', ')+'</p><form method="post" action="/consent"><input type="hidden" name="handle" value="'+escapeHtml(handle)+'"><button name="decision" value="allow">Continue to sign in</button><button name="decision" value="deny">Cancel</button></form></main></html>';
}
