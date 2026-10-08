import {DurableObject} from 'cloudflare:workers';
import type {AuthorizationGrant,ConnectionGrant,PolicyResult,RequestEnvelope,VerifiedActor} from './contracts.ts';
import {authorizeRequest} from './request-policy.ts';

interface RegistryState {
  version:1; connection:ConnectionGrant|null; authorizations:AuthorizationGrant[];
  entitlement:{expiresAt:number;version:number};
}
/** Connect Free default; operators raise it per deployment with DAILY_TOOL_CALL_LIMIT. */
export const DEFAULT_DAILY_TOOL_CALL_LIMIT=50;
interface DailyUsage {day:string;count:number;}
function dailyLimit(raw:unknown):number{const value=Number(raw);return Number.isSafeInteger(value)&&value>0?value:DEFAULT_DAILY_TOOL_CALL_LIMIT;}
const tools=new Set(['recall','search','fetch','get_status','remember','session_init','close_session','report_feedback','report_outcome']);
const scopes=new Set(['slm:read','slm:write','slm:session']);
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
      await this.commit({...this.state,authorizations:[...this.state.authorizations,a]});return {value:undefined};
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
  async provisionAccess(owner:string,expiresAt:number):Promise<void>{
    if(!Number.isSafeInteger(expiresAt)||expiresAt<=Date.now())throw new Error('invalid_entitlement');
    await this.mutation(async()=>{
      if(this.state.connection?.ownerId!==owner||this.state.connection.revokedAt!==null)return {error:'connection_unavailable'};
      if(this.state.entitlement.version>0)return this.state.entitlement.expiresAt===expiresAt?{value:undefined}:{error:'entitlement_conflict'};
      await this.commit({...this.state,entitlement:{version:1,expiresAt}});return {value:undefined};
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
      const stored=this.usage??await this.ctx.storage.get<DailyUsage>('usage-day')??null;
      const current=stored&&stored.day===day?stored:{day,count:0};
      if(current.count>=dailyLimit(this.env.DAILY_TOOL_CALL_LIMIT)){this.usage=current;return {allowed:false,code:'DAILY_LIMIT_REACHED',httpStatus:429};}
      const next={day,count:current.count+1};
      await this.ctx.storage.put('usage-day',next);this.usage=next;
      return decision;
    });
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
