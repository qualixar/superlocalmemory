import {DurableObject} from 'cloudflare:workers';
import type {AuthorizationGrant,ConnectionGrant,PolicyResult,RequestEnvelope,VerifiedActor} from './contracts.ts';
import {GRANT_SCOPE_ORDER} from './grant.ts';
import {authorizeRequest,TOOL_SCOPES} from './request-policy.ts';

interface RegistryState {
  version:1; connection:ConnectionGrant|null; authorizations:AuthorizationGrant[];
  entitlement:{expiresAt:number;version:number};
}
/** Connect Free default; operators raise it per deployment with DAILY_TOOL_CALL_LIMIT. */
export const DEFAULT_DAILY_TOOL_CALL_LIMIT=50;
interface DailyUsage {day:string;count:number;}
/** Registry-stamped app facts for the owner's Connected apps list (null = predates tracking). */
interface AppMeta {createdAt:number|null;lastUsedAt:number|null;}
export interface ConnectedApp {authorizationId:string;clientId:string;consentedScopes:string[];authorizationVersion:number;createdAt:number|null;lastUsedAt:number|null;}
/** Last-use is coarse on purpose: at most one extra write per app per minute. */
const LAST_USED_RESOLUTION_MS=60000;
function dailyLimit(raw:unknown):number{const value=Number(raw);return Number.isSafeInteger(value)&&value>0?value:DEFAULT_DAILY_TOOL_CALL_LIMIT;}
/** Mesh polling (inbox, wait) has its own daily budget per connection; sending is capped per app. */
export const DEFAULT_MESH_POLL_DAILY_LIMIT=2000;
export const MESH_SEND_DAILY_LIMIT=200;
interface MeshUsage {day:string;polls:number;sends:Record<string,number>;}
function pollLimit(raw:unknown):number{const value=Number(raw);return Number.isSafeInteger(value)&&value>0?value:DEFAULT_MESH_POLL_DAILY_LIMIT;}
const tools:ReadonlySet<string>=new Set(TOOL_SCOPES.keys());
const scopes:ReadonlySet<string>=new Set(GRANT_SCOPE_ORDER);
const id=(value:unknown):value is string=>typeof value==='string'&&/^[A-Za-z0-9_.:-]{1,256}$/.test(value);
const client=(value:unknown):value is string=>typeof value==='string'&&value.length>0&&value.length<=2048&&value.trim()===value&&!/[\x00-\x1f\x7f]/.test(value);
const version=(value:unknown):value is number=>Number.isSafeInteger(value)&&(value as number)>=1&&(value as number)<Number.MAX_SAFE_INTEGER;
function list(value:unknown,allowed:ReadonlySet<string>):boolean {
  return Array.isArray(value)&&value.length<=32&&new Set(value).size===value.length&&value.every(x=>typeof x==='string'&&allowed.has(x));
}
function validConnection(c:ConnectionGrant):boolean {
  if(!c||![c.ownerId,c.connectionId,c.installationId,c.profileId].every(id)||!version(c.policyVersion)||c.revokedAt!==null||!list(c.allowedTools,tools))return false;
  if(![c.allowCorrection,c.allowSharedRead,c.allowGlobalRead].every(x=>typeof x==='boolean'))return false;
  return !!c.origin&&c.origin.kind==='relay'&&c.origin.installationId===c.installationId&&c.origin.profileId===c.profileId;
}
function validAuthorization(a:AuthorizationGrant):boolean {
  if(!a||![a.ownerId,a.authorizationId,a.connectionId].every(id)||!client(a.clientId)||!version(a.authorizationVersion)||a.revokedAt!==null||!list(a.consentedTools,tools)||!list(a.consentedScopes,scopes))return false;
  if(![a.consentedCorrection,a.consentedSharedRead,a.consentedGlobalRead].every(x=>typeof x==='boolean'))return false;
  if(a.providerGrantRef&&![a.providerGrantRef.userId,a.providerGrantRef.grantId].every(id))return false;
  return a.audience==='https://mcp.superlocalmemory.com/mcp';
}
function validActor(a:VerifiedActor):boolean {
  return !!a&&[a.ownerId,a.authorizationId,a.connectionId].every(id)&&client(a.clientId)&&list(a.scopes,scopes)&&['oauth','service'].includes(a.credentialKind);
}

/** Authoritative per-connection registry. Only internal Worker RPC can mutate
 * it. No memory bodies, origin secrets or OAuth token values are persisted.
 * Durable revocation precedes success; provider cleanup is a separate retry.
 */
export class RegistryDO extends DurableObject<Record<string,unknown>> {
  private state:RegistryState={version:1,connection:null,authorizations:[],entitlement:{expiresAt:0,version:0}};
  /** Separate key: the strictly versioned registry-state schema is unchanged. */
  private usage:DailyUsage|null=null;
  private meta:Record<string,AppMeta>|null=null;
  private mesh:MeshUsage|null=null;
  private async meshUsage(day:string):Promise<MeshUsage>{
    const stored=this.mesh??await this.ctx.storage.get<MeshUsage>('mesh-usage')??null;
    return stored&&stored.day===day?stored:{day,polls:0,sends:{}};
  }
  private async appMeta():Promise<Record<string,AppMeta>>{this.meta??=(await this.ctx.storage.get<Record<string,AppMeta>>('authorization-meta'))??{};return this.meta;}
  constructor(ctx:DurableObjectState,env:Record<string,unknown>){
    super(ctx,env);
    this.ctx.blockConcurrencyWhile(async()=>{
      const stored=await this.ctx.storage.get<RegistryState>('registry-state');
      if(stored){if(stored.version!==1)throw new Error('unsupported_registry');this.state=stored;}
    });
  }
  private async commit(next:RegistryState):Promise<void>{await this.ctx.storage.put('registry-state',next);this.state=next;}
  private async mutation<T>(operation:()=>Promise<{value:T}|{error:string}>):Promise<T>{
    const result=await this.ctx.blockConcurrencyWhile(operation);
    if('error' in result)throw new Error(result.error);
    return result.value;
  }
  async configure(connection:ConnectionGrant):Promise<void>{
    if(!validConnection(connection))throw new Error('invalid_connection');
    // Pick contract fields, excluding accidental operator credentials/extra data.
    const c:ConnectionGrant={connectionId:connection.connectionId,ownerId:connection.ownerId,installationId:connection.installationId,profileId:connection.profileId,origin:{kind:'relay',installationId:connection.installationId,profileId:connection.profileId},allowedTools:[...connection.allowedTools],allowCorrection:connection.allowCorrection,allowSharedRead:connection.allowSharedRead,allowGlobalRead:connection.allowGlobalRead,policyVersion:connection.policyVersion,revokedAt:null};
    await this.mutation(async()=>{
      if(this.state.connection?.revokedAt)return {error:'connection_revoked'};
      if(this.state.connection&&JSON.stringify(this.state.connection)!==JSON.stringify(c))return {error:'connection_conflict'};
      await this.commit({...this.state,connection:c});return {value:undefined};
    });
  }
  async addAuthorization(authorization:AuthorizationGrant):Promise<void>{
    if(!validAuthorization(authorization))throw new Error('invalid_authorization');
    const a:AuthorizationGrant={authorizationId:authorization.authorizationId,audience:authorization.audience,ownerId:authorization.ownerId,clientId:authorization.clientId,connectionId:authorization.connectionId,consentedTools:[...authorization.consentedTools],consentedScopes:[...authorization.consentedScopes],consentedCorrection:authorization.consentedCorrection,consentedSharedRead:authorization.consentedSharedRead,consentedGlobalRead:authorization.consentedGlobalRead,authorizationVersion:authorization.authorizationVersion,revokedAt:null,...(authorization.providerGrantRef?{providerGrantRef:{...authorization.providerGrantRef}}:{})};
    await this.mutation(async()=>{
      const c=this.state.connection;
      if(!c||c.revokedAt!==null)return {error:'connection_unavailable'};
      if(a.ownerId!==c.ownerId||a.connectionId!==c.connectionId)return {error:'binding_mismatch'};
      const existing=this.state.authorizations.find(x=>x.authorizationId===a.authorizationId);
      if(existing)return JSON.stringify(existing)===JSON.stringify(a)?{value:undefined}:{error:'authorization_conflict'};
      if(this.state.authorizations.length>=256)return {error:'capacity_exhausted'};
      await this.commit({...this.state,authorizations:[...this.state.authorizations,a]});
      const meta={...await this.appMeta(),[a.authorizationId]:{createdAt:Date.now(),lastUsedAt:null}};
      await this.ctx.storage.put('authorization-meta',meta);this.meta=meta;
      return {value:undefined};
    });
  }
  async setEntitlement(owner:string,expiresAt:number,expectedVersion:number):Promise<number>{
    if(!Number.isSafeInteger(expiresAt)||expiresAt<0)throw new Error('invalid_entitlement');
    return this.mutation<number>(async()=>{
      if(this.state.connection?.ownerId!==owner)return {error:'owner_mismatch'};
      if(this.state.connection.revokedAt!==null)return {error:'connection_revoked'};
      if(this.state.entitlement.version!==expectedVersion||!Number.isSafeInteger(expectedVersion)||expectedVersion>=Number.MAX_SAFE_INTEGER)return {error:'version_conflict'};
      const entitlement={expiresAt,version:expectedVersion+1};await this.commit({...this.state,entitlement});return {value:entitlement.version};
    });
  }
  /** Access follows the laptop credential. It only ever moves forward: a renewed
   * credential extends it, and a repeated or older provisioning call changes nothing. */
  async provisionAccess(owner:string,expiresAt:number):Promise<void>{
    if(!Number.isSafeInteger(expiresAt)||expiresAt<=Date.now())throw new Error('invalid_entitlement');
    await this.mutation(async()=>{
      if(this.state.connection?.ownerId!==owner||this.state.connection.revokedAt!==null)return {error:'connection_unavailable'};
      const current=this.state.entitlement;
      if(current.version>0&&current.expiresAt>=expiresAt)return {value:undefined};
      if(current.version>=Number.MAX_SAFE_INTEGER)return {error:'version_conflict'};
      await this.commit({...this.state,entitlement:{version:current.version+1,expiresAt}});return {value:undefined};
    });
  }
  async admit(actor:VerifiedActor,resource:string,request:RequestEnvelope):Promise<PolicyResult>{
    if(!validActor(actor))return {allowed:false,code:'INVALID_PRINCIPAL',httpStatus:401};
    // Snapshot and decision are synchronous after the input gate has opened.
    const state=this.state;
    if(state.entitlement.expiresAt<=Date.now())return {allowed:false,code:'ENTITLEMENT_REQUIRED',httpStatus:403};
    const authorization=state.authorizations.find(a=>a.authorizationId===actor.authorizationId)??null;
    const decision=authorizeRequest(actor,authorization,state.connection,resource,request);
    // Only real work counts: a client's initialize/tools/list handshake is free.
    if(!decision.allowed||request.rpcMethod!=='tools/call')return decision;
    return this.ctx.blockConcurrencyWhile(async():Promise<PolicyResult>=>{
      const day=new Date().toISOString().slice(0,10);
      if(request.toolName==='mesh_inbox'||request.toolName==='mesh_wait')return this.admitPoll(day,decision);
      const stored=this.usage??await this.ctx.storage.get<DailyUsage>('usage-day')??null;
      const current=stored&&stored.day===day?stored:{day,count:0};
      if(current.count>=dailyLimit(this.env.DAILY_TOOL_CALL_LIMIT)){this.usage=current;return {allowed:false,code:'DAILY_LIMIT_REACHED',httpStatus:429};}
      const sending=request.toolName==='mesh_send';const mesh=sending?await this.meshUsage(day):null;
      if(mesh&&(mesh.sends[actor.authorizationId]??0)>=MESH_SEND_DAILY_LIMIT){this.mesh=mesh;return {allowed:false,code:'MESH_SEND_LIMIT',httpStatus:429};}
      const next={day,count:current.count+1};const now=Date.now();
      if(mesh){const updated={...mesh,sends:{...mesh.sends,[actor.authorizationId]:(mesh.sends[actor.authorizationId]??0)+1}};await this.ctx.storage.put('mesh-usage',updated);this.mesh=updated;}
      const meta=await this.appMeta();const known=meta[actor.authorizationId];
      if(known?.lastUsedAt!=null&&now-known.lastUsedAt<LAST_USED_RESOLUTION_MS){await this.ctx.storage.put('usage-day',next);}
      else{const updated={...meta,[actor.authorizationId]:{createdAt:known?.createdAt??null,lastUsedAt:now}};await this.ctx.storage.put({'usage-day':next,'authorization-meta':updated});this.meta=updated;}
      this.usage=next;
      return decision;
    });
  }
  /** Polling the inbox is the web app's idle loop, so it is budgeted apart from the daily tool-call allowance. */
  private async admitPoll(day:string,decision:PolicyResult):Promise<PolicyResult>{
    const mesh=await this.meshUsage(day);
    if(mesh.polls>=pollLimit(this.env.MESH_POLL_DAILY_LIMIT)){this.mesh=mesh;return {allowed:false,code:'DAILY_LIMIT_REACHED',httpStatus:429};}
    const updated={...mesh,polls:mesh.polls+1};await this.ctx.storage.put('mesh-usage',updated);this.mesh=updated;
    return decision;
  }
  /** May a one-time upload link still be used? Only while the connection and the owner's access are live and some app
   * still holds the consent that lets it make one (writing, pictures and the upload tool). Answers no on anything unclear. */
  async uploadsAllowed():Promise<boolean>{
    const {connection,authorizations,entitlement}=this.state;
    if(!connection||connection.revokedAt!==null||entitlement.expiresAt<=Date.now())return false;
    return authorizations.some(a=>a.revokedAt===null&&a.consentedScopes.includes('slm:write')&&a.consentedScopes.includes('slm:media')&&a.consentedTools.includes('media_upload_link'));
  }
  /** Active grants for the owner's Connected apps list. No tokens or memory data. */
  async listAuthorizations(owner:string):Promise<ConnectedApp[]>{
    if(!this.state.connection||this.state.connection.ownerId!==owner)throw new Error('owner_mismatch');
    const meta=await this.appMeta();
    return this.state.authorizations.filter(a=>a.revokedAt===null).map(a=>({authorizationId:a.authorizationId,clientId:a.clientId,consentedScopes:[...a.consentedScopes],authorizationVersion:a.authorizationVersion,createdAt:meta[a.authorizationId]?.createdAt??null,lastUsedAt:meta[a.authorizationId]?.lastUsedAt??null}));
  }
  async revokeAuthorization(owner:string,identifier:string,expectedVersion:number):Promise<number>{
    return this.mutation<number>(async()=>{
      if(this.state.connection?.ownerId!==owner)return {error:'owner_mismatch'};
      const a=this.state.authorizations.find(x=>x.authorizationId===identifier);
      if(!a)return {error:'not_found'};
      if(a.revokedAt!==null)return {value:a.authorizationVersion};
      if(a.authorizationVersion!==expectedVersion||!version(expectedVersion))return {error:'version_conflict'};
      const revoked={...a,revokedAt:new Date().toISOString(),authorizationVersion:a.authorizationVersion+1};
      await this.commit({...this.state,authorizations:this.state.authorizations.map(x=>x.authorizationId===identifier?revoked:x)});return {value:revoked.authorizationVersion};
    });
  }
  async revokeConnection(owner:string,expectedVersion:number):Promise<number>{
    return this.mutation<number>(async()=>{
      const c=this.state.connection;
      if(!c||c.ownerId!==owner)return {error:'owner_mismatch'};
      if(c.revokedAt!==null)return {value:c.policyVersion};
      if(c.policyVersion!==expectedVersion||!version(expectedVersion))return {error:'version_conflict'};
      const revoked={...c,revokedAt:new Date().toISOString(),policyVersion:c.policyVersion+1};
      await this.commit({...this.state,connection:revoked});return {value:revoked.policyVersion};
    });
  }
  async fetch(_request:Request):Promise<Response>{return new Response('not_found',{status:404});}
}
