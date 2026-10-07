import {CompactEncrypt,compactDecrypt} from 'jose';
import {validateIndexedToken,type IssuerEnv} from './issuer-protocol.ts';
import {verifyDeviceProof} from './device-proof.ts';
import {tokenDigest} from './issued-token-protocol.ts';
import type {NativeAuthProps} from './authorization-server.ts';
import type {ConnectEnv} from './worker-connect.ts';
import type {OwnedConnection} from './owner-index-do.ts';
export interface OwnerControlEnv extends IssuerEnv,ConnectEnv {DEVICE_WRAP_KEY:string;}
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
  const token=crypto.randomUUID().replaceAll('-','')+crypto.randomUUID().replaceAll('-','');const expiresAtMs=Date.now()+30*24*3600*1000;
  const value:Delivery={device_token:token,expires_at_ms:expiresAtMs,generation:1};
  const credentialEnvelope=await new CompactEncrypt(new TextEncoder().encode(JSON.stringify(value))).setProtectedHeader({alg:'dir',enc:'A256GCM'}).encrypt(wrapKey(env));
  const candidate:OwnedConnection={connectionId:props.connectionId,installationId:props.installationId,profileId:props.profileId,host:bootstrap.host,permissions:bootstrap.permissions,credentialEnvelope,deviceDigest:await tokenDigest(token),deviceJkt:props.deviceJkt,deviceExpiresAtMs:expiresAtMs,generation:1,revokedAt:null,cleanupPending:false};
  try{await owner.addConnection(props.ownerId,props.installationId,props.profileId,clientId,candidate);}catch{
   // A concurrent exact enrollment may have won. Read authoritative identity;
   // other failures stay unavailable, never replace an existing credential.
   row=await owner.getConnection(props.ownerId,props.connectionId);if(!row)throw new Error('provision_unavailable');
  }
  row=await owner.getConnection(props.ownerId,props.connectionId);
 }
 if(!row||row.revokedAt!==null||row.installationId!==props.installationId||row.profileId!==props.profileId||row.deviceJkt!==props.deviceJkt)throw new Error('connection_unavailable');
 const registry=env.REGISTRIES.getByName(props.connectionId);const allowedTools=['recall','search','fetch','get_status',...(row.permissions.write?['remember']:[]),...(row.permissions.session?['session_init','close_session','report_feedback','report_outcome']:[])];
 await registry.configure({connectionId:props.connectionId,ownerId:props.ownerId,installationId:props.installationId,profileId:props.profileId,origin:{kind:'relay',installationId:props.installationId,profileId:props.profileId},allowedTools,allowCorrection:false,allowSharedRead:false,allowGlobalRead:false,policyVersion:1,revokedAt:null});
 await env.DEVICES.getByName(row.deviceDigest).configure({ownerId:props.ownerId,connectionId:props.connectionId,installationId:props.installationId,profileId:props.profileId,deviceDigest:row.deviceDigest,deviceJkt:props.deviceJkt,expiresAtMs:row.deviceExpiresAtMs});
 await env.RELAYS.getByName(props.connectionId).configureBinding({ownerId:props.ownerId,connectionId:props.connectionId,installationId:props.installationId,profileId:props.profileId,deviceDigest:row.deviceDigest,deviceExpiresAt:row.deviceExpiresAtMs});
 await registry.provisionAccess(props.ownerId,row.deviceExpiresAtMs);
 return delivery(row,env);
}
export async function ownerControlFetch(request:Request,env:OwnerControlEnv,_ctx:ExecutionContext):Promise<Response>{
 const url=new URL(request.url);if(url.origin!=='https://auth.superlocalmemory.com'||!['/owner/connections','/owner/revoke','/owner/verify'].includes(url.pathname))return result(404,{error:'not_found'});
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
  if(url.pathname==='/owner/verify'){
   const row=await owner.getConnection(props.ownerId,props.connectionId);
   if(!row||row.revokedAt!==null||row.installationId!==props.installationId||row.profileId!==props.profileId)return result(403,{error:'connection_unavailable'});
   for(const message of [
    {jsonrpc:'2.0',id:1,method:'initialize',params:{protocolVersion:'2025-06-18',capabilities:{},clientInfo:{name:'SuperLocalMemory Connection Check',version:'4.1.23'}}},
    {jsonrpc:'2.0',id:2,method:'tools/call',params:{name:'get_status',arguments:{}}}
   ]){
    const bytes=new TextEncoder().encode(JSON.stringify(message));
    const reply=await env.RELAYS.getByName(props.connectionId).forwardCurrent({v:1,kind:'request',id:crypto.randomUUID(),deadlineAt:Date.now()+5000,headers:[['content-type','application/json'],['accept','application/json'],['mcp-protocol-version','2025-06-18']],bodyBase64:btoa(String.fromCharCode(...bytes))});
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
