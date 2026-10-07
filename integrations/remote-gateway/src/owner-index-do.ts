import {DurableObject} from 'cloudflare:workers';
import {validPermissions,type RemotePermissions} from './bootstrap-do.ts';
interface NativeBinding {installationId:string;profileId:string;clientId:string;deviceJkt:string;}
export interface OwnedConnection {connectionId:string;installationId:string;profileId:string;host:string;permissions:RemotePermissions;credentialEnvelope:string;deviceDigest:string;deviceJkt:string;deviceExpiresAtMs:number;generation:number;revokedAt:string|null;cleanupPending:boolean;}
interface OwnerState {ownerId:string|null;native:NativeBinding[];connections:OwnedConnection[];cancelledIds:string[];}
const id=(v:unknown):v is string=>typeof v==='string'&&/^[-A-Za-z0-9_]{1,128}$/.test(v);
const client=(v:unknown):v is string=>typeof v==='string'&&v.length>0&&v.length<=2048&&!/[\x00-\x1f\x7f]/.test(v);
const jkt=(v:unknown):v is string=>typeof v==='string'&&/^[-A-Za-z0-9_]{43}$/.test(v);
/** Authoritative owner/installation/profile directory. Metadata and encrypted
 * operational credentials only. Invocation is through private Worker RPC.
 */
export class OwnerIndexDO extends DurableObject<Record<string,unknown>> {
 private state:OwnerState={ownerId:null,native:[],connections:[],cancelledIds:[]};
 constructor(ctx:DurableObjectState,env:Record<string,unknown>){super(ctx,env);this.ctx.blockConcurrencyWhile(async()=>{const saved=await ctx.storage.get<OwnerState>('owner');if(saved)this.state={...this.state,...saved};});}
 private async commit(next:OwnerState):Promise<void>{await this.ctx.storage.put('owner',next);this.state=next;}
 async bind(ownerId:string,installationId:string,profileId:string,clientId:string,deviceJkt:string):Promise<void>{
  if(!/^[0-9]{1,32}$/.test(ownerId)||!id(installationId)||!id(profileId)||!client(clientId)||!jkt(deviceJkt))throw new Error('invalid_native_binding');
  const error=await this.ctx.blockConcurrencyWhile(async()=>{
   if(this.state.ownerId!==null&&this.state.ownerId!==ownerId)return 'owner_mismatch';
   const existing=this.state.native.find(x=>x.installationId===installationId&&x.profileId===profileId);
   if(existing)return existing.clientId===clientId&&existing.deviceJkt===deviceJkt?null:'native_binding_conflict';
   if(this.state.native.length>=128)return 'native_capacity_exhausted';
   await this.commit({...this.state,ownerId,native:[...this.state.native,{installationId,profileId,clientId,deviceJkt}]});return null;
  });if(error)throw new Error(error);
 }
 async authorizeNative(ownerId:string,installationId:string,profileId:string,clientId:string,deviceJkt:string):Promise<boolean>{
  return this.state.ownerId===ownerId&&this.state.native.some(x=>x.installationId===installationId&&x.profileId===profileId&&x.clientId===clientId&&x.deviceJkt===deviceJkt);
 }
 async addConnection(ownerId:string,installationId:string,profileId:string,clientId:string,connection:OwnedConnection):Promise<void>{
  if(!connection||!/^[a-f0-9]{32}$/.test(connection.connectionId)||connection.installationId!==installationId||connection.profileId!==profileId||!validPermissions(connection.permissions)||!['muse','chatgpt','claude_web','claude_code_web','composio','other_mcp'].includes(connection.host)||typeof connection.credentialEnvelope!=='string'||connection.credentialEnvelope.length>8192||!/^[a-f0-9]{64}$/.test(connection.deviceDigest)||!jkt(connection.deviceJkt)||!Number.isSafeInteger(connection.deviceExpiresAtMs)||connection.deviceExpiresAtMs<=Date.now()||!Number.isSafeInteger(connection.generation)||connection.generation<1||connection.revokedAt!==null||connection.cleanupPending!==false)throw new Error('invalid_owned_connection');
  const row:OwnedConnection={connectionId:connection.connectionId,installationId,profileId,host:connection.host,permissions:{...connection.permissions},credentialEnvelope:connection.credentialEnvelope,deviceDigest:connection.deviceDigest,deviceJkt:connection.deviceJkt,deviceExpiresAtMs:connection.deviceExpiresAtMs,generation:connection.generation,revokedAt:null,cleanupPending:false};
  const error=await this.ctx.blockConcurrencyWhile(async()=>{
   // No public RPC method await inside the closed input gate: evaluate the
   // already-loaded binding synchronously to avoid self-call deadlock.
   if(this.state.cancelledIds.includes(row.connectionId))return 'connection_cancelled';
   if(this.state.ownerId!==ownerId||!this.state.native.some(x=>x.installationId===installationId&&x.profileId===profileId&&x.clientId===clientId&&x.deviceJkt===row.deviceJkt))return 'native_binding_mismatch';
   const old=this.state.connections.find(x=>x.connectionId===row.connectionId);
   if(old)return JSON.stringify(old)===JSON.stringify(row)?null:'connection_conflict';
   if(this.state.connections.length>=128||this.state.connections.filter(x=>x.revokedAt===null).length>=32)return 'connection_capacity_exhausted';
   await this.commit({...this.state,connections:[...this.state.connections,row]});return null;
  });if(error)throw new Error(error);
 }
 async cancelConnection(ownerId:string,connectionId:string):Promise<void>{
  if(!/^[0-9]{1,32}$/.test(ownerId)||!/^[a-f0-9]{32}$/.test(connectionId))throw new Error('invalid_cancellation');
  const error=await this.ctx.blockConcurrencyWhile(async()=>{
   if(this.state.ownerId!==null&&this.state.ownerId!==ownerId)return 'owner_mismatch';
   if(this.state.cancelledIds.includes(connectionId))return null;
   if(this.state.cancelledIds.length>=256)return 'cancellation_capacity_exhausted';
   await this.commit({...this.state,ownerId,cancelledIds:[...this.state.cancelledIds,connectionId]});return null;
  });if(error)throw new Error(error);
 }
 async list(ownerId:string):Promise<Omit<OwnedConnection,'credentialEnvelope'|'deviceDigest'|'deviceJkt'>[]>{
  if(this.state.ownerId!==ownerId)return [];
  return this.state.connections.map(({credentialEnvelope:_e,deviceDigest:_d,deviceJkt:_j,...row})=>structuredClone(row));
 }
 async getConnection(ownerId:string,connectionId:string):Promise<OwnedConnection|null>{
  if(this.state.ownerId!==ownerId)return null;
  const row=this.state.connections.find(x=>x.connectionId===connectionId);return row?structuredClone(row):null;
 }
 async revoke(ownerId:string,connectionId:string,expectedGeneration:number):Promise<number>{
  const result=await this.ctx.blockConcurrencyWhile(async():Promise<{value:number}|{error:string}>=>{
   if(this.state.ownerId!==ownerId)return {error:'owner_mismatch'};
   const row=this.state.connections.find(x=>x.connectionId===connectionId);if(!row)return {error:'not_found'};
   if(row.revokedAt!==null)return {value:row.generation};
   if(row.generation!==expectedGeneration||!Number.isSafeInteger(expectedGeneration)||expectedGeneration>=Number.MAX_SAFE_INTEGER)return {error:'version_conflict'};
   const next={...row,revokedAt:new Date().toISOString(),generation:row.generation+1,cleanupPending:true};
   await this.commit({...this.state,connections:this.state.connections.map(x=>x.connectionId===connectionId?next:x)});return {value:next.generation};
  });if('error' in result)throw new Error(result.error);return result.value;
 }
 async markCleanup(ownerId:string,connectionId:string,generation:number):Promise<boolean>{
  return this.ctx.blockConcurrencyWhile(async()=>{
   if(this.state.ownerId!==ownerId)return false;
   const row=this.state.connections.find(x=>x.connectionId===connectionId);if(!row||row.revokedAt===null||row.generation!==generation)return false;
   await this.commit({...this.state,connections:this.state.connections.map(x=>x.connectionId===connectionId?{...x,cleanupPending:false}:x)});return true;
  });
 }
 async fetch(_request:Request):Promise<Response>{return new Response('not_found',{status:404});}
}
