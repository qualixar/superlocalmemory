import {queueProviderCleanup,retryProviderCleanup} from './provider-cleanup.ts';
import {readAuthorizationBody} from './authorization-body.ts';
import {authorizationServer,type NativeAuthProps} from './authorization-server.ts';
import {indexedToken,recordIssuedTokens,type IssuedTokenEnv} from './issued-token-protocol.ts';
import {AUTH_ISSUER,MCP_RESOURCE,OWNER_RESOURCE} from './authorization-policy.ts';
import type {AuthWorkerEnv} from './worker-auth.ts';
import type {IndexedToken} from './token-index-do.ts';
import type {AuthProps} from './contracts.ts';
export type IssuerEnv=AuthWorkerEnv&IssuedTokenEnv;
let nextCleanupAt=0;
function failed(status:number,error:string){return Response.json({error},{status,headers:{'Cache-Control':'no-store'}});}
async function denyAuthorization(ref:IndexedToken,env:IssuerEnv):Promise<void>{
 if(ref.audience===MCP_RESOURCE)await env.REGISTRIES.getByName(ref.connectionId).revokeAuthorization(ref.ownerId,ref.authorizationId,1);
 else if(ref.audience===OWNER_RESOURCE){
  const owner=env.OWNERS.getByName(ref.ownerId);await owner.cancelConnection(ref.ownerId,ref.connectionId);const row=await owner.getConnection(ref.ownerId,ref.connectionId);
  if(row){const generation=await owner.revoke(ref.ownerId,ref.connectionId,row.generation);
   await env.REGISTRIES.getByName(ref.connectionId).revokeConnection(ref.ownerId,1);
   await env.DEVICES.getByName(row.deviceDigest).revoke(ref.ownerId);await env.RELAYS.getByName(ref.connectionId).revoke();
   await owner.markCleanup(ref.ownerId,ref.connectionId,generation);
  }
 }
 await env.TOKEN_INDEX.getByName(ref.tokenDigest.slice(0,4)).revoke(ref.tokenDigest,ref.clientId);
}
function submittedClient(request:Request,fields:URLSearchParams):string|null {
 const auth=request.headers.get('Authorization');if(auth?.startsWith('Basic ')){try{const decoded=atob(auth.slice(6));const separator=decoded.indexOf(':');return separator>=0?decodeURIComponent(decoded.slice(0,separator)):null;}catch{return null;}}
 return fields.get('client_id');
}
/** Holds successful token/revocation acknowledgements until durable state commits.
 * Protocol parsing and credential validation remain owned by the maintained SDK. */
export async function issuerProtocol(request:Request,env:IssuerEnv,ctx:ExecutionContext):Promise<Response>{
 if(new URL(request.url).pathname!=='/oauth/token'||request.method!=='POST')return authorizationServer.fetch(request,env,ctx);
 if(Date.now()>=nextCleanupAt){nextCleanupAt=Date.now()+60000;ctx.waitUntil(retryProviderCleanup(env,authorizationServer.getOAuthApi(env)));}
 if(request.headers.get('Content-Type')?.split(';')[0]!=='application/x-www-form-urlencoded')return authorizationServer.fetch(request,env,ctx);
 let raw:string;try{raw=await readAuthorizationBody(request);}catch(error){return failed(error instanceof Error&&error.message==='body_timeout'?408:413,'invalid_request');}
 request=new Request(request.url,{method:request.method,headers:request.headers,body:raw,signal:request.signal});
 const fields=new URLSearchParams(raw);const refresh=fields.get('grant_type')==='refresh_token';const revoke=!fields.has('grant_type')&&fields.has('token');
 const lookupValue=refresh?fields.get('refresh_token'):revoke?fields.get('token'):null;
 let ref:IndexedToken|null=null;
 if(lookupValue){ref=await indexedToken(lookupValue,env);if(refresh&&(!ref||ref.revoked||ref.tokenKind!=='refresh'))return failed(400,'invalid_grant');}
 const response=await authorizationServer.fetch(request,env,ctx);if(!response.ok)return response;
 if(revoke){
  if(ref&&ref.clientId===submittedClient(request,fields)){
   try{await denyAuthorization(ref,env);}catch{return failed(503,'temporarily_unavailable');}
  }
  return response;
 }
 const body=await response.clone().json() as {access_token?:string;refresh_token?:string};if(!body.access_token)return response;
 const api=authorizationServer.getOAuthApi(env);
 try{const issued=await recordIssuedTokens(body,env,api,refresh?ref!.validUntil:undefined);
  if(ref&&(issued.grant.clientId!==ref.clientId||issued.userId!==ref.ownerId||issued.audience!==ref.audience||issued.grant.props.connectionId!==ref.connectionId))throw new Error('binding_mismatch');
  return response;
 }catch{
  // Hold token delivery; await cleanup or persist only provider grant references.
  try{const issued=await api.unwrapToken(body.access_token);if(issued){try{await api.revokeGrant(issued.grantId,issued.userId);}catch{await queueProviderCleanup(env,{grantId:issued.grantId,userId:issued.userId});}}}catch{console.warn('provider_cleanup_recovery_unavailable');}
  return failed(503,'temporarily_unavailable');
 }
}
export async function validateIndexedToken(resource:string,token:string,env:IssuerEnv){
 if(![MCP_RESOURCE,OWNER_RESOURCE].includes(resource))return null;
 const ref=await indexedToken(token,env);if(!ref||ref.revoked||ref.tokenKind!=='access'||ref.audience!==resource)return null;
 const valid=await authorizationServer.validateToken<AuthProps|NativeAuthProps>(resource,token,env);
 if(!valid||valid.userId!==ref.ownerId||valid.clientId!==ref.clientId||valid.props.ownerId!==ref.ownerId||valid.props.connectionId!==ref.connectionId)return null;
 if(resource===MCP_RESOURCE){
  if(!('authorizationId' in valid.props)||valid.props.authorizationId!==ref.authorizationId)return null;
  const current=await env.REGISTRIES.getByName(ref.connectionId).admit({ownerId:ref.ownerId,clientId:ref.clientId,connectionId:ref.connectionId,authorizationId:ref.authorizationId,audience:resource,credentialKind:'oauth',scopes:valid.scope as ('slm:read'|'slm:write'|'slm:session')[]},resource,{era:'legacy',rpcMethod:'initialize',rpcId:1,originalBody:new Uint8Array()});
  if(!current.allowed)return null;
 }
 return valid;
}
