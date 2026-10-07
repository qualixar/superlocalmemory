import {anonymousAdmission,type SetupAdmissionEnv} from './anonymous-admission.ts';
import {registrationDiagnostic} from './dcr-diagnostics.ts';
import {readAuthorizationBody} from './authorization-body.ts';
import {ownerControlFetch} from './owner-control.ts';
import type {ConnectEnv} from './worker-connect.ts';
import {issuerProtocol,validateIndexedToken} from './issuer-protocol.ts';
import type {IssuedTokenEnv} from './issued-token-protocol.ts';
import {WorkerEntrypoint} from 'cloudflare:workers';
import type {AuthRequest} from '@cloudflare/workers-oauth-provider';
import {calculateJwkThumbprint} from 'jose';
import {authorizationServer,type AuthorizationEnv,type NativeAuthProps} from './authorization-server.ts';
import {AUTH_ISSUER,MCP_RESOURCE,OWNER_RESOURCE,selectedScopes,validateAuthorizationRequest,memoryAuthorizationRequest,verifyGithubIdentity} from './authorization-policy.ts';
import {escapeHtml,exchangeGithubCode,githubAuthorizationUrl,renderConsentPage,renderAuthPage,renderAuthFailure} from './auth-flow.ts';
import {tokenHash} from './device-proof.ts';
import type {BootstrapBinding} from './bootstrap-do.ts';
import type {AuthProps,Scope} from './contracts.ts';
export interface AuthWorkerEnv extends AuthorizationEnv,IssuedTokenEnv,ConnectEnv,SetupAdmissionEnv {GITHUB_CLIENT_ID:string;GITHUB_CLIENT_SECRET:string;DEVICE_WRAP_KEY:string;DCR_DIAGNOSTICS?:string;}
interface ConsentContext {request:AuthRequest;bootstrapId?:string;ownerId?:string;}
function response(status:number,error:string):Response{return Response.json({error},{status,headers:{'Cache-Control':'no-store'}});}
function html(body:string,headers:Headers,redirectUri?:string,status=200):Response{
 const nonce=crypto.randomUUID().replaceAll('-','');
 let destination='';
 if(redirectUri){try{const target=new URL(redirectUri);if(!target.username&&!target.password&&(target.protocol==='https:'||target.protocol==='http:'&&['127.0.0.1','localhost','[::1]'].includes(target.hostname)))destination=' '+target.origin;}catch{/* No additional destination for malformed metadata. */}}
 headers.set('Content-Type','text/html;charset=utf-8');headers.set('Cache-Control','no-store');headers.set('Referrer-Policy','strict-origin');
 // Chrome applies form-action to the redirect chain. Only the identity
 // provider and this SDK-validated client's return origin are allowed.
 headers.set('Content-Security-Policy',"default-src 'none'; style-src 'nonce-"+nonce+"'; form-action 'self' https://github.com"+destination+"; frame-ancestors 'none'; base-uri 'none'");
 headers.set('X-Content-Type-Options','nosniff');
 return new Response(body.replace('<style>','<style nonce="'+nonce+'">'),{status,headers});
}
function interactiveFailure(request:Request,status:number,error:string):Response {
 return request.headers.get('Accept')?.includes('text/html')?html(renderAuthFailure(error),new Headers(),undefined,status):response(status,error);
}
function redirect(location:string,headers=new Headers()):Response {headers.set('Location',location);headers.set('Cache-Control','no-store');return new Response(null,{status:302,headers});}
const boundedText=readAuthorizationBody;

async function form(request:Request):Promise<URLSearchParams>{
 if(request.headers.get('Origin')!==AUTH_ISSUER||request.headers.get('Content-Type')?.split(';')[0]!=='application/x-www-form-urlencoded')throw new Error('invalid_form');
 return new URLSearchParams(await boundedText(request));
}
async function beginConsent(request:AuthRequest,env:AuthWorkerEnv,context:Omit<ConsentContext,'request'>={}):Promise<Response>{
 const api=authorizationServer.getOAuthApi(env);const consent=await api.beginConsent(request);
 await env.OAUTH_KV.put('slm-consent:'+consent.handle,JSON.stringify({request,...context}),{expirationTtl:900});
 const description=await api.describeConsent(request);
 let profileId:string|undefined;
 if(context.bootstrapId){const row=await env.BOOTSTRAPS.getByName(context.bootstrapId).get();if(!row)return html(renderAuthFailure('connection_unavailable'),new Headers(),undefined,404);profileId=row.profileId;}
 const page=renderConsentPage(consent.handle,{...description,redirectHostname:description.redirectHost,profileId});
 return html(page,consent.headers,request.redirectUri);
}
async function consentContext(handle:string,env:AuthWorkerEnv):Promise<ConsentContext|null>{
 if(!handle||handle.length>512)return null;
 return env.OAUTH_KV.get<ConsentContext>('slm-consent:'+handle,'json');
}
async function handleConsent(request:Request,env:AuthWorkerEnv):Promise<Response>{
 const fields=await form(request);const handle=fields.get('handle')??'';const context=await consentContext(handle,env);if(!context)return interactiveFailure(request,400,'consent_unavailable');
 const api=authorizationServer.getOAuthApi(env);
 if(fields.get('decision')==='deny'){const denied=await api.denyConsent(request,handle);await env.OAUTH_KV.delete('slm-consent:'+handle);return redirect(denied.redirectTo,denied.headers);}
 if(fields.get('decision')!=='allow')return response(400,'invalid_decision');
 const scopes=context.request.resource===OWNER_RESOURCE?['slm:connect']:context.request.scope.filter(s=>s==='slm:read'||s==='slm:write'&&fields.get('write')==='yes'||s==='slm:session'&&fields.get('session')==='yes');
 const approved=await api.approveConsent(request,handle,{scope:scopes});await env.OAUTH_KV.delete('slm-consent:'+handle);
 const verifier=crypto.randomUUID().replaceAll('-','')+crypto.randomUUID().replaceAll('-','');
 const upstream=await api.beginUpstream(approved.request,{data:{verifier,bootstrapId:context.bootstrapId},headers:approved.headers});
 return redirect(githubAuthorizationUrl(env.GITHUB_CLIENT_ID,upstream.state,await tokenHash(verifier)),upstream.headers);
}
async function githubCallback(request:Request,env:AuthWorkerEnv):Promise<Response>{
 const api=authorizationServer.getOAuthApi(env);const resumed=await api.finishUpstream<{verifier:string;bootstrapId?:string}>(request);
 const code=new URL(request.url).searchParams.get('code')??'';
 const token=await exchangeGithubCode(env.GITHUB_CLIENT_ID,env.GITHUB_CLIENT_SECRET,code,resumed.data.verifier);
 const ownerId=await verifyGithubIdentity(token);
 if(resumed.request.resource===OWNER_RESOURCE){
  const connectionId=resumed.data.bootstrapId;if(!connectionId)return response(400,'connection_unavailable');
  const bootstrap=env.BOOTSTRAPS.getByName(connectionId);const row=await bootstrap.get();
  if(!row||row.authRequest.clientId!==resumed.request.clientId||row.authRequest.codeChallenge!==resumed.request.codeChallenge||row.status==='cancelled')return response(400,'connection_unavailable');
  const deviceJkt=await calculateJwkThumbprint(row.deviceJwk,'sha256');
  await bootstrap.approve(ownerId);await env.OWNERS.getByName(ownerId).bind(ownerId,row.installationId,row.profileId,resumed.request.clientId,deviceJkt);await bootstrap.confirm(ownerId,resumed.request.clientId);
  const props:NativeAuthProps={kind:'native',ownerId,installationId:row.installationId,profileId:row.profileId,connectionId,deviceJkt};
  const completed=await api.completeAuthorization({request:resumed.request,userId:ownerId,metadata:{kind:'native'},scope:['slm:connect'],props,revokeExistingGrants:false});
  return redirect(completed.redirectTo,resumed.headers);
 }
 if(resumed.request.resource!==MCP_RESOURCE)return response(400,'invalid_resource');
 const connections=(await env.OWNERS.getByName(ownerId).list(ownerId)).filter(c=>c.revokedAt===null);
 if(!connections.length)return html(renderAuthPage('Link your computer first','<h1>Link your computer first</h1><p>No active computer connection is available for this GitHub account. Open your local SLM dashboard, enable a web connection, then reconnect this application using the same GitHub account.</p>'),resumed.headers,resumed.request.redirectUri);
 const consent=await api.beginConsent(resumed.request);
 await env.OAUTH_KV.put('slm-consent:'+consent.handle,JSON.stringify({request:resumed.request,ownerId}),{expirationTtl:900});
 const options=connections.map(c=>'<option value="'+escapeHtml(c.connectionId)+'">'+escapeHtml(c.profileId)+' ('+escapeHtml(c.host)+')</option>').join('');
 return html(renderAuthPage('Choose your SLM connection','<h1>Choose your SLM connection</h1><p>GitHub sign-in is complete. Select the local profile this application may use.</p><form method="post" action="/select"><input type="hidden" name="handle" value="'+escapeHtml(consent.handle)+'"><label>Local profile <select name="connection_id">'+options+'</select></label><div class="actions"><button name="decision" value="allow">Connect</button><button name="decision" value="deny">Cancel</button></div></form><details><summary>Approved permissions</summary><p>'+resumed.request.scope.map(escapeHtml).join(', ')+'</p></details>'),consent.headers,resumed.request.redirectUri);
}
async function selectConnection(request:Request,env:AuthWorkerEnv):Promise<Response>{
 const fields=await form(request);const handle=fields.get('handle')??'';const saved=await consentContext(handle,env);if(!saved?.ownerId)return interactiveFailure(request,400,'consent_unavailable');
 const api=authorizationServer.getOAuthApi(env);
 if(fields.get('decision')==='deny'){const result=await api.denyConsent(request,handle);await env.OAUTH_KV.delete('slm-consent:'+handle);return redirect(result.redirectTo,result.headers);}
 if(fields.get('decision')!=='allow')return response(400,'invalid_decision');
 const connectionId=fields.get('connection_id')??'';if(!/^[a-f0-9]{32}$/.test(connectionId))return response(400,'connection_unavailable');
 const connection=await env.OWNERS.getByName(saved.ownerId).getConnection(saved.ownerId,connectionId);
 if(!connection||connection.revokedAt!==null)return response(403,'connection_unavailable');
 const scopes=selectedScopes(saved.request.scope,connection.permissions);if(!scopes.length)return response(403,'insufficient_scope');
 const approved=await api.approveConsent(request,handle,{scope:scopes});await env.OAUTH_KV.delete('slm-consent:'+handle);
 if(approved.request.clientId!==saved.request.clientId||approved.request.resource!==MCP_RESOURCE)return response(400,'consent_unavailable');
 const authorizationId=crypto.randomUUID();const props:AuthProps={ownerId:saved.ownerId,authorizationId,connectionId};
 const tools=['recall','search','fetch','get_status',...(scopes.includes('slm:write')?['remember']:[]),...(scopes.includes('slm:session')?['session_init','close_session','report_feedback','report_outcome']:[])];
 await env.REGISTRIES.getByName(connectionId).addAuthorization({authorizationId,ownerId:saved.ownerId,connectionId,clientId:approved.request.clientId,audience:MCP_RESOURCE,consentedTools:tools,consentedScopes:scopes as Scope[],consentedCorrection:false,consentedSharedRead:false,consentedGlobalRead:false,authorizationVersion:1,revokedAt:null});
 const completed=await api.completeAuthorization({request:approved.request,userId:saved.ownerId,metadata:{connectionId},scope:scopes,props,revokeExistingGrants:false});
 return redirect(completed.redirectTo,approved.headers);
}
export async function authFetch(request:Request,env:AuthWorkerEnv,ctx:ExecutionContext):Promise<Response>{
 const url=new URL(request.url);if(url.origin!==AUTH_ISSUER)return response(403,'host_denied');
 const limited=await anonymousAdmission(request,env);if(limited)return limited;
 try{
  if(url.pathname==='/oauth/register'&&request.method==='POST'&&env.DCR_DIAGNOSTICS==='1'){
   let raw:string;try{raw=await readAuthorizationBody(request,{limit:1048576});}catch{return response(400,'invalid_request');}
   let metadata:unknown;try{metadata=JSON.parse(raw);}catch{metadata=null;}
   const copy=new Request(request.url,{method:request.method,headers:request.headers,body:raw});
   const reply=await issuerProtocol(copy,env,ctx);let outcome:unknown;try{outcome=await reply.clone().json();}catch{outcome=null;}
   const diagnostic=registrationDiagnostic(metadata,reply.status,outcome);
   ctx.waitUntil(env.OAUTH_KV.put('slm-dcr-diagnostic:'+Date.now()+':'+crypto.randomUUID(),JSON.stringify(diagnostic),{expirationTtl:600}).catch(()=>{console.warn('dcr_diagnostic_unavailable');}));
   return reply;
  }

  if(url.pathname==='/authorize'&&request.method==='GET'){
   const parsed=memoryAuthorizationRequest(await authorizationServer.getOAuthApi(env).parseAuthRequest(request));
   if(!parsed)return response(400,'invalid_authorization');
   return await beginConsent(parsed,env);
  }
  if(url.pathname==='/owner-login'&&request.method==='GET'){
   const id=url.searchParams.get('connection_id')??'';if(!/^[a-f0-9]{32}$/.test(id))return interactiveFailure(request,404,'connection_unavailable');
   const row=await env.BOOTSTRAPS.getByName(id).get();if(!row||!['pending','approved'].includes(row.status)||!validateAuthorizationRequest(row.authRequest))return interactiveFailure(request,404,'connection_unavailable');
   return await beginConsent(row.authRequest,env,{bootstrapId:id});
  }
  if(url.pathname==='/bootstrap/cancel'&&request.method==='POST'){
   if(request.headers.get('Origin')!==null)return response(403,'origin_denied');
   const body=JSON.parse(await boundedText(request)) as {connection_id?:unknown;verifier?:unknown};
   if(typeof body.connection_id!=='string'||!/^[a-f0-9]{32}$/.test(body.connection_id)||typeof body.verifier!=='string')return response(400,'invalid_bootstrap');
   const snapshot=await env.BOOTSTRAPS.getByName(body.connection_id).cancel(body.verifier);
   if(snapshot.ownerId){
    const owner=env.OWNERS.getByName(snapshot.ownerId);await owner.cancelConnection(snapshot.ownerId,body.connection_id);const row=await owner.getConnection(snapshot.ownerId,body.connection_id);
    if(row){
     const generation=await owner.revoke(snapshot.ownerId,body.connection_id,row.generation);
     await env.REGISTRIES.getByName(body.connection_id).revokeConnection(snapshot.ownerId,1);
     await env.DEVICES.getByName(row.deviceDigest).revoke(snapshot.ownerId);await env.RELAYS.getByName(body.connection_id).revoke();
     await owner.markCleanup(snapshot.ownerId,body.connection_id,generation);
    }
   }
   return Response.json({cancelled:true,connection_id:body.connection_id},{headers:{'Cache-Control':'no-store'}});
  }
  if(url.pathname==='/bootstrap'&&request.method==='POST'){
   if(request.headers.get('Origin')!==null)return response(403,'origin_denied');
   const body=JSON.parse(await boundedText(request)) as BootstrapBinding&{authorizationUrl?:string};
   if(typeof body.authorizationUrl!=='string'||typeof body.connectionId!=='string'||!/^[a-f0-9]{32}$/.test(body.connectionId))return response(400,'invalid_bootstrap');
   const parsed=await authorizationServer.getOAuthApi(env).parseAuthRequest(new Request(body.authorizationUrl));
   if(!validateAuthorizationRequest(parsed)||parsed.resource!==OWNER_RESOURCE)return response(400,'invalid_bootstrap');
   await env.BOOTSTRAPS.getByName(body.connectionId).configure({...body,authRequest:parsed});
   return Response.json({connection_id:body.connectionId,authorize_url:AUTH_ISSUER+'/owner-login?connection_id='+body.connectionId},{status:201,headers:{'Cache-Control':'no-store'}});
  }
  if(['/owner/connections','/owner/revoke','/owner/verify'].includes(url.pathname))return await ownerControlFetch(request,env,ctx);
  if(url.pathname==='/consent'&&request.method==='POST')return await handleConsent(request,env);
  if(url.pathname==='/select'&&request.method==='POST')return await selectConnection(request,env);
  if(url.pathname==='/github/callback'&&request.method==='GET')return await githubCallback(request,env);
  return await issuerProtocol(request,env,ctx);
 }catch{return ['/owner-login','/authorize','/consent','/select','/github/callback'].includes(url.pathname)?interactiveFailure(request,400,'authorization_unavailable'):response(400,'authorization_unavailable');}
}
export default class AuthWorker extends WorkerEntrypoint<AuthWorkerEnv>{
 fetch(request:Request):Promise<Response>{return authFetch(request,this.env,this.ctx);}
 validateToken(resource:string,token:string){return validateIndexedToken(resource,token,this.env);}
}
