import {CompactEncrypt,compactDecrypt} from 'jose';
import {validateIndexedToken,type IssuerEnv} from './issuer-protocol.ts';
import {verifyDeviceProof} from './device-proof.ts';
import {tokenDigest} from './issued-token-protocol.ts';
import {RELAY_DEADLINE_MS} from './relay-protocol.ts';
import {DEVICE_CREDENTIAL_TTL_MS,RENEWAL_WINDOW_MS} from './credential-lifetime.ts';
import {authorizationServer,type AuthorizationEnv,type NativeAuthProps} from './authorization-server.ts';
import {readAuthorizationBody} from './authorization-body.ts';
import type {ConnectEnv} from './worker-connect.ts';
import type {OwnedConnection} from './owner-index-do.ts';
export interface OwnerControlEnv extends IssuerEnv,ConnectEnv,AuthorizationEnv {DEVICE_WRAP_KEY:string;}
/** Client names are third-party input: plain text only, bounded, no control characters. */
function displayName(value:unknown):string {
 const text=typeof value==='string'?value.replace(/[\u0000-\u001f\u007f-\u009f]/g,'').replace(/\s+/g,' ').trim():'';
 return (text||'Unnamed app').slice(0,80);
}
function displayHost(uris:readonly string[]|undefined):string|null {
 try{return uris&&uris[0]?new URL(uris[0]).hostname.slice(0,253):null;}catch{return null;}
}
/** The owner's Connected apps list: names come from the client's own registration. */
async function connectedApps(props:NativeAuthProps,env:OwnerControlEnv,extended:boolean):Promise<Response>{
 const apps=await env.REGISTRIES.getByName(props.connectionId).listAuthorizations(props.ownerId);
 const api=authorizationServer.getOAuthApi(env);
 const listed=await Promise.all(apps.map(async app=>{
  const client=await api.lookupClient(app.clientId).catch(()=>null);
  return {authorization_id:app.authorizationId,name:displayName(client?.clientName),client_host:displayHost(client?.redirectUris),
   permissions:{read:app.consentedScopes.includes('slm:read'),save:app.consentedScopes.includes('slm:write'),session:app.consentedScopes.includes('slm:session'),...(extended?{mesh:app.consentedScopes.includes('slm:mesh'),media:app.consentedScopes.includes('slm:media')}:{})},
   version:app.authorizationVersion,connected_at_ms:app.createdAt,last_used_at_ms:app.lastUsedAt};
 }));
 return result(200,{connection_id:props.connectionId,apps:listed});
}
/** Only a body of exactly {"version":2} asks for the longer permission list; everything else gets the shape older laptops validate. */
async function wantsExtendedList(request:Request):Promise<boolean>{
 try{const body=JSON.parse(await readAuthorizationBody(request,{limit:64})) as unknown;return !!body&&typeof body==='object'&&!Array.isArray(body)&&Object.keys(body).length===1&&(body as {version?:unknown}).version===2;}catch{return false;}
}
/** A fresh key for the signed per-request grant. It is returned here once and never readable again. */
async function grantKey(request:Request,props:NativeAuthProps,env:OwnerControlEnv):Promise<Response>{
 let body:unknown;
 try{body=JSON.parse(await readAuthorizationBody(request,{limit:64}));}catch{return result(400,{error:'invalid_request'});}
 if(!body||typeof body!=='object'||Array.isArray(body)||Object.keys(body).length!==0)return result(400,{error:'invalid_request'});
 try{
  const minted=await env.RELAYS.getByName(props.connectionId).rotateGrantKey(props.ownerId);
  return result(200,{version:minted.version,key:minted.key,connection_id:props.connectionId});
 }catch(error){
  const code=error instanceof Error?error.message:'';
  if(code==='grant_unavailable')return result(503,{error:'grant_unavailable'});
  if(['connection_revoked','connection_unconfigured','owner_mismatch'].includes(code))return result(403,{error:'connection_unavailable'});
  throw error;
 }
}
async function removeApp(request:Request,props:NativeAuthProps,env:OwnerControlEnv):Promise<Response>{
 let body:unknown;
 try{body=JSON.parse(await readAuthorizationBody(request,{limit:1024}));}catch{return result(400,{error:'invalid_request'});}
 const value=body as {authorization_id?:unknown;expected_version?:unknown};
 if(!body||typeof body!=='object'||Object.keys(body).length!==2||typeof value.authorization_id!=='string'||!/^[A-Za-z0-9_.:-]{1,256}$/.test(value.authorization_id)||!Number.isSafeInteger(value.expected_version)||(value.expected_version as number)<1)return result(400,{error:'invalid_request'});
 try{
  const version=await env.REGISTRIES.getByName(props.connectionId).revokeAuthorization(props.ownerId,value.authorization_id,value.expected_version as number);
  return result(200,{revoked:true,version});
 }catch(error){
  const code=error instanceof Error?error.message:'';
  if(code==='not_found')return result(404,{error:'not_found'});
  if(code==='version_conflict')return result(409,{error:'version_conflict'});
  if(code==='owner_mismatch')return result(403,{error:'owner_mismatch'});
  throw error;
 }
}
/** The only owner operations; the auth Worker routes exactly these here. */
export const OWNER_CONTROL_PATHS:readonly string[]=['/owner/connections','/owner/revoke','/owner/verify','/owner/apps','/owner/apps/revoke','/owner/renew','/owner/grant-key'];
function result(status:number,value:unknown):Response{return Response.json(value,{status,headers:{'Cache-Control':'no-store'}});}
function wrapKey(env:OwnerControlEnv):Uint8Array {if(!/^[a-f0-9]{64}$/.test(env.DEVICE_WRAP_KEY))throw new Error('credential_wrap_unavailable');return new Uint8Array(env.DEVICE_WRAP_KEY.match(/../g)!.map(x=>parseInt(x,16)));}
interface Delivery {device_token:string;expires_at_ms:number;generation:number;}
async function delivery(row:OwnedConnection,env:OwnerControlEnv):Promise<Delivery>{const decoded=await compactDecrypt(row.credentialEnvelope,wrapKey(env),{keyManagementAlgorithms:['dir'],contentEncryptionAlgorithms:['A256GCM']});const value=JSON.parse(new TextDecoder().decode(decoded.plaintext)) as Delivery;if(typeof value.device_token!=='string'||await tokenDigest(value.device_token)!==row.deviceDigest||value.expires_at_ms!==row.deviceExpiresAtMs||value.generation!==row.generation)throw new Error('credential_delivery_unavailable');return value;}
async function provision(props:NativeAuthProps,clientId:string,env:OwnerControlEnv):Promise<Delivery>{
 const owner=env.OWNERS.getByName(props.ownerId);let row=await owner.getConnection(props.ownerId,props.connectionId);
 if(row?.revokedAt!==null&&row!==null)throw new Error('connection_revoked');
 if(!row){
  const bootstrap=await env.BOOTSTRAPS.getByName(props.connectionId).get();
  if(!bootstrap||bootstrap.status!=='completed'||bootstrap.ownerId!==props.ownerId||bootstrap.authRequest.clientId!==clientId||bootstrap.installationId!==props.installationId||bootstrap.profileId!==props.profileId)throw new Error('bootstrap_unavailable');
  const issued=await issueCredential(1,env);
  const candidate:OwnedConnection={connectionId:props.connectionId,installationId:props.installationId,profileId:props.profileId,host:bootstrap.host,permissions:bootstrap.permissions,credentialEnvelope:issued.credentialEnvelope,deviceDigest:issued.deviceDigest,deviceJkt:props.deviceJkt,deviceExpiresAtMs:issued.deviceExpiresAtMs,generation:1,revokedAt:null,cleanupPending:false};
  try{await owner.addConnection(props.ownerId,props.installationId,props.profileId,clientId,candidate);}catch{
   // A concurrent exact enrollment may have won. Read authoritative identity;
   // other failures stay unavailable, never replace an existing credential.
   row=await owner.getConnection(props.ownerId,props.connectionId);if(!row)throw new Error('provision_unavailable');
  }
  row=await owner.getConnection(props.ownerId,props.connectionId);
 }
 if(!row||row.revokedAt!==null||row.installationId!==props.installationId||row.profileId!==props.profileId||row.deviceJkt!==props.deviceJkt)throw new Error('connection_unavailable');
 // Binding the current credential is idempotent, so provisioning also completes a
 // renewal that stopped after the owner record changed.
 await bindCredential(props,row,env);
 return delivery(row,env);
}
async function issueCredential(generation:number,env:OwnerControlEnv):Promise<{credentialEnvelope:string;deviceDigest:string;deviceExpiresAtMs:number}>{
 const token=crypto.randomUUID().replaceAll('-','')+crypto.randomUUID().replaceAll('-','');const deviceExpiresAtMs=Date.now()+DEVICE_CREDENTIAL_TTL_MS;
 const value:Delivery={device_token:token,expires_at_ms:deviceExpiresAtMs,generation};
 const credentialEnvelope=await new CompactEncrypt(new TextEncoder().encode(JSON.stringify(value))).setProtectedHeader({alg:'dir',enc:'A256GCM'}).encrypt(wrapKey(env));
 return {credentialEnvelope,deviceDigest:await tokenDigest(token),deviceExpiresAtMs};
}
async function bindCredential(props:NativeAuthProps,row:OwnedConnection,env:OwnerControlEnv):Promise<void>{
 const registry=env.REGISTRIES.getByName(props.connectionId);const allowedTools=['recall','search','fetch','get_status',...(row.permissions.write?['remember']:[]),...(row.permissions.session?['session_init','close_session','report_feedback','report_outcome']:[])];
 await registry.configure({connectionId:props.connectionId,ownerId:props.ownerId,installationId:props.installationId,profileId:props.profileId,origin:{kind:'relay',installationId:props.installationId,profileId:props.profileId},allowedTools,allowCorrection:false,allowSharedRead:false,allowGlobalRead:false,policyVersion:1,revokedAt:null});
 await env.DEVICES.getByName(row.deviceDigest).configure({ownerId:props.ownerId,connectionId:props.connectionId,installationId:props.installationId,profileId:props.profileId,deviceDigest:row.deviceDigest,deviceJkt:props.deviceJkt,expiresAtMs:row.deviceExpiresAtMs});
 await env.RELAYS.getByName(props.connectionId).configureBinding({ownerId:props.ownerId,connectionId:props.connectionId,installationId:props.installationId,profileId:props.profileId,deviceDigest:row.deviceDigest,deviceExpiresAt:row.deviceExpiresAtMs});
 await registry.provisionAccess(props.ownerId,row.deviceExpiresAtMs);
}
/** The laptop replaces its credential in the second half of its life. The owner
 * record changes first, so the old credential stops matching at once; then the
 * device, relay and access are bound to the new one. */
async function renew(request:Request,props:NativeAuthProps,env:OwnerControlEnv):Promise<Response>{
 let body:unknown;
 try{body=JSON.parse(await readAuthorizationBody(request,{limit:256}));}catch{return result(400,{error:'invalid_request'});}
 const expected=(body as {expected_generation?:unknown})?.expected_generation;
 if(!body||typeof body!=='object'||Array.isArray(body)||Object.keys(body).length!==1||!Number.isSafeInteger(expected)||(expected as number)<1)return result(400,{error:'invalid_request'});
 const owner=env.OWNERS.getByName(props.ownerId);const row=await owner.getConnection(props.ownerId,props.connectionId);
 if(!row||row.revokedAt!==null||row.installationId!==props.installationId||row.profileId!==props.profileId||row.deviceJkt!==props.deviceJkt)return result(403,{error:'connection_unavailable'});
 if(row.generation!==expected)return result(409,{error:'version_conflict'});
 if(row.deviceExpiresAtMs-Date.now()>RENEWAL_WINDOW_MS)return result(409,{error:'renewal_not_due'});
 const issued=await issueCredential((expected as number)+1,env);
 let rotated:OwnedConnection;
 try{rotated=await owner.rotateCredential(props.ownerId,props.connectionId,expected as number,issued);}catch(error){
  const code=error instanceof Error?error.message:'';
  if(code==='version_conflict')return result(409,{error:'version_conflict'});
  if(code==='connection_revoked'||code==='not_found'||code==='owner_mismatch')return result(403,{error:'connection_unavailable'});
  throw error;
 }
 await bindCredential(props,rotated,env);
 // The owner record no longer matches the old credential, so the connector Worker
 // already refuses it; retiring its device record as well is housekeeping.
 await env.DEVICES.getByName(row.deviceDigest).revoke(props.ownerId).catch(()=>undefined);
 return result(200,{...await delivery(rotated,env),connection_id:props.connectionId,profile_id:props.profileId});
}
export async function ownerControlFetch(request:Request,env:OwnerControlEnv,_ctx:ExecutionContext):Promise<Response>{
 const url=new URL(request.url);if(url.origin!=='https://auth.superlocalmemory.com'||!OWNER_CONTROL_PATHS.includes(url.pathname))return result(404,{error:'not_found'});
 if(request.method!=='POST')return result(405,{error:'method_not_allowed'});
 if(request.headers.get('Origin')!==null)return result(403,{error:'origin_denied'});
 const token=/^Bearer ([^\s]+)$/.exec(request.headers.get('Authorization')??'')?.[1];const proof=request.headers.get('DPoP');if(!token||!proof)return result(401,{error:'owner_unauthorized'});
 try{
  const principal=await validateIndexedToken('https://auth.superlocalmemory.com/owner',token,env);
  if(!principal||!('kind' in principal.props)||principal.props.kind!=='native'||!principal.scope.includes('slm:connect'))return result(401,{error:'owner_unauthorized'});
  const props=principal.props;const owner=env.OWNERS.getByName(props.ownerId);if(!await owner.authorizeNative(props.ownerId,props.installationId,props.profileId,principal.clientId,props.deviceJkt))return result(401,{error:'owner_unauthorized'});
  const verified=await verifyDeviceProof(proof,{jkt:props.deviceJkt,method:'POST',url:request.url,token});
  const digest=await tokenDigest(token);const replay=env.DEVICES.getByName('control:'+digest);
  await replay.configure({ownerId:props.ownerId,connectionId:props.connectionId,installationId:props.installationId,profileId:props.profileId,deviceDigest:digest,deviceJkt:props.deviceJkt,expiresAtMs:principal.expiresAt*1000});
  if(!await replay.consume(digest,verified))return result(401,{error:'owner_unauthorized'});
  if(url.pathname==='/owner/apps'||url.pathname==='/owner/apps/revoke'||url.pathname==='/owner/grant-key'){
   const row=await owner.getConnection(props.ownerId,props.connectionId);
   if(!row||row.revokedAt!==null||row.installationId!==props.installationId||row.profileId!==props.profileId)return result(403,{error:'connection_unavailable'});
   if(url.pathname==='/owner/grant-key')return await grantKey(request,props,env);
   return url.pathname==='/owner/apps'?await connectedApps(props,env,await wantsExtendedList(request)):await removeApp(request,props,env);
  }
  // Awaited inside the try, so a failure becomes 503 rather than escaping as a sign-in error.
  if(url.pathname==='/owner/renew')return await renew(request,props,env);
  if(url.pathname==='/owner/verify'){
   const row=await owner.getConnection(props.ownerId,props.connectionId);
   if(!row||row.revokedAt!==null||row.installationId!==props.installationId||row.profileId!==props.profileId)return result(403,{error:'connection_unavailable'});
   for(const message of [
    {jsonrpc:'2.0',id:1,method:'initialize',params:{protocolVersion:'2025-06-18',capabilities:{},clientInfo:{name:'SuperLocalMemory Connection Check',version:'4.1.23'}}},
    {jsonrpc:'2.0',id:2,method:'tools/call',params:{name:'get_status',arguments:{}}}
   ]){
    const bytes=new TextEncoder().encode(JSON.stringify(message));
    const reply=await env.RELAYS.getByName(props.connectionId).forwardCurrent({v:1,kind:'request',id:crypto.randomUUID(),deadlineAt:Date.now()+RELAY_DEADLINE_MS,headers:[['content-type','application/json'],['accept','application/json'],['mcp-protocol-version','2025-06-18']],bodyBase64:btoa(String.fromCharCode(...bytes))});
    if(!reply.ok)return result(503,{error:'verification_unavailable'});
    const body=await reply.json() as {jsonrpc?:unknown;id?:unknown;error?:unknown;result?:{isError?:unknown}};
    if(body.jsonrpc!=='2.0'||body.id!==message.id||body.error!==undefined||!body.result||body.result.isError===true)return result(503,{error:'verification_unavailable'});
   }
   return result(200,{verified:true,connection_id:props.connectionId});
  }
  if(url.pathname==='/owner/revoke'){
   const row=await owner.getConnection(props.ownerId,props.connectionId);if(!row)return result(404,{error:'not_found'});
   await owner.cancelConnection(props.ownerId,props.connectionId);
   const generation=await owner.revoke(props.ownerId,props.connectionId,row.generation);
   await env.REGISTRIES.getByName(props.connectionId).revokeConnection(props.ownerId,1);
   await env.DEVICES.getByName(row.deviceDigest).revoke(props.ownerId);await env.RELAYS.getByName(props.connectionId).revoke();
   await owner.markCleanup(props.ownerId,props.connectionId,generation);return result(200,{revoked:true,generation});
  }
  const value=await provision(props,principal.clientId,env);return result(200,{...value,connection_id:props.connectionId,profile_id:props.profileId});
 }catch{return result(503,{error:'owner_operation_unavailable'});}
}
