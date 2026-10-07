import {DurableObject} from 'cloudflare:workers';
import type {AuthRequest} from '@cloudflare/workers-oauth-provider';
import {publicDeviceKey,tokenHash} from './device-proof.ts';
import type {JWK} from 'jose';
export interface RemotePermissions {read:boolean;write:boolean;correction:boolean;session:boolean;}
export interface BootstrapBinding {connectionId:string;installationId:string;profileId:string;host:string;permissions:RemotePermissions;authRequest:AuthRequest;deviceJwk:JWK;expiresAtMs:number;}
export interface BootstrapRecord extends BootstrapBinding {status:'pending'|'approved'|'completed'|'cancelled';version:number;ownerId:string|null;}
export function validPermissions(p:RemotePermissions):boolean{return !!p&&Object.keys(p).length===4&&Object.values(p).every(v=>typeof v==='boolean')&&p.read===true&&(!p.correction||p.write);}
function valid(binding:BootstrapBinding):boolean {
 if(!binding||typeof binding.installationId!=='string'||typeof binding.profileId!=='string'||typeof binding.connectionId!=='string'||!/^[-A-Za-z0-9_]{1,128}$/.test(binding.installationId)||!/^[-A-Za-z0-9_]{1,64}$/.test(binding.profileId)||!/^[a-f0-9]{32}$/.test(binding.connectionId)||!['muse','chatgpt','claude_web','claude_code_web','composio','other_mcp'].includes(binding.host)||!validPermissions(binding.permissions))return false;
 const auth=binding.authRequest;
 if(!auth||typeof auth.clientId!=='string'||!auth.clientId||auth.clientId.length>2048||auth.resource!=='https://auth.superlocalmemory.com/owner'||auth.issuer!=='https://auth.superlocalmemory.com'||auth.responseType!=='code'||auth.codeChallengeMethod!=='S256'||typeof auth.codeChallenge!=='string'||!/^[-A-Za-z0-9_]{43}$/.test(auth.codeChallenge)||typeof auth.state!=='string'||!/^[-A-Za-z0-9_]{32,128}$/.test(auth.state)||!Array.isArray(auth.scope)||auth.scope.length!==1||auth.scope[0]!=='slm:connect')return false;
 try{const redirect=new URL(auth.redirectUri);if(redirect.protocol!=='http:'||redirect.hostname!=='127.0.0.1'||redirect.pathname!=='/api/v3/connections/callback'||redirect.search||redirect.hash||redirect.username||redirect.password)return false;publicDeviceKey(binding.deviceJwk);}catch{return false;}
 return Number.isSafeInteger(binding.expiresAtMs)&&binding.expiresAtMs>Date.now()&&binding.expiresAtMs<=Date.now()+15*60*1000;
}
/** One native PKCE bootstrap. Public IDs confer no control authority. */
export class BootstrapDO extends DurableObject<Record<string,unknown>> {
 private record:BootstrapRecord|null=null;
 constructor(ctx:DurableObjectState,env:Record<string,unknown>){super(ctx,env);this.ctx.blockConcurrencyWhile(async()=>{this.record=await ctx.storage.get<BootstrapRecord>('bootstrap')??null;});}
 async configure(binding:BootstrapBinding):Promise<void>{
  if(!valid(binding))throw new Error('invalid_bootstrap');
  const curated:BootstrapRecord={connectionId:binding.connectionId,installationId:binding.installationId,profileId:binding.profileId,host:binding.host,permissions:{...binding.permissions},authRequest:{...binding.authRequest,scope:[...binding.authRequest.scope]},deviceJwk:publicDeviceKey(binding.deviceJwk),expiresAtMs:binding.expiresAtMs,status:'pending',version:1,ownerId:null};
  const error=await this.ctx.blockConcurrencyWhile(async()=>{
   if(this.record?.status==='cancelled')return 'bootstrap_cancelled';
   if(this.record){const current={...this.record,status:'pending',version:1,ownerId:null};return JSON.stringify(current)===JSON.stringify(curated)?null:'bootstrap_conflict';}
   await this.ctx.storage.put('bootstrap',curated);this.record=curated;return null;
  });if(error)throw new Error(error);
 }
 async get():Promise<BootstrapRecord|null>{return this.record&&this.record.expiresAtMs>Date.now()?structuredClone(this.record):null;}
 async approve(ownerId:string):Promise<void>{
  if(!/^[0-9]{1,32}$/.test(ownerId))throw new Error('invalid_owner');
  const error=await this.ctx.blockConcurrencyWhile(async()=>{
   const row=this.record;if(!row||row.expiresAtMs<=Date.now()||row.status==='cancelled')return 'bootstrap_unavailable';
   if(row.ownerId!==null&&row.ownerId!==ownerId)return 'owner_mismatch';
   if(row.status!=='pending')return null;
   const next={...row,ownerId,status:'approved' as const,version:row.version+1};await this.ctx.storage.put('bootstrap',next);this.record=next;return null;
  });if(error)throw new Error(error);
 }
 async confirm(ownerId:string,clientId:string):Promise<void>{
  const error=await this.ctx.blockConcurrencyWhile(async()=>{
   const row=this.record;if(!row||row.expiresAtMs<=Date.now()||row.ownerId!==ownerId||row.authRequest.clientId!==clientId||!['approved','completed'].includes(row.status))return 'bootstrap_unavailable';
   if(row.status==='completed')return null;
   const next={...row,status:'completed' as const,version:row.version+1};await this.ctx.storage.put('bootstrap',next);this.record=next;return null;
  });if(error)throw new Error(error);
 }
 async cancel(verifier:string):Promise<BootstrapRecord>{
  if(typeof verifier!=='string'||!/^[-A-Za-z0-9_.~]{43,128}$/.test(verifier))throw new Error('invalid_bootstrap_proof');
  const challenge=await tokenHash(verifier);
  const error=await this.ctx.blockConcurrencyWhile(async()=>{
   if(!this.record||this.record.authRequest.codeChallenge!==challenge)return 'invalid_bootstrap_proof';
   if(this.record.status==='cancelled')return null;
   const next={...this.record,status:'cancelled' as const,version:this.record.version+1};await this.ctx.storage.put('bootstrap',next);this.record=next;return null;
  });if(error)throw new Error(error);
  return structuredClone(this.record!);
 }
 async fetch(_request:Request):Promise<Response>{return new Response('not_found',{status:404});}
}
