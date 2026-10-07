import {DurableObject} from 'cloudflare:workers';
import type {VerifiedDeviceProof} from './device-proof.ts';
export interface DeviceBinding {ownerId:string;connectionId:string;installationId:string;profileId:string;deviceDigest:string;deviceJkt:string;expiresAtMs:number;}
interface State {binding:DeviceBinding|null;revoked:boolean;}
function valid(binding:DeviceBinding):boolean{return !!binding&&typeof binding.ownerId==='string'&&/^[0-9]{1,32}$/.test(binding.ownerId)&&/^[a-f0-9]{32}$/.test(binding.connectionId)&&[binding.installationId,binding.profileId].every(x=>typeof x==='string'&&/^[-A-Za-z0-9_]{1,128}$/.test(x))&&/^[a-f0-9]{64}$/.test(binding.deviceDigest)&&/^[-A-Za-z0-9_]{43}$/.test(binding.deviceJkt)&&Number.isSafeInteger(binding.expiresAtMs)&&binding.expiresAtMs>Date.now();}
/** Per-device digest binding and durable bounded proof replay authority. */
export class DeviceIndexDO extends DurableObject<Record<string,unknown>>{
 private state:State={binding:null,revoked:false};
 constructor(ctx:DurableObjectState,env:Record<string,unknown>){super(ctx,env);this.ctx.blockConcurrencyWhile(async()=>{this.state=await ctx.storage.get<State>('device')??this.state;});}
 async configure(binding:DeviceBinding):Promise<void>{
  if(!valid(binding))throw new Error('invalid_device_binding');
  const row:DeviceBinding={ownerId:binding.ownerId,connectionId:binding.connectionId,installationId:binding.installationId,profileId:binding.profileId,deviceDigest:binding.deviceDigest,deviceJkt:binding.deviceJkt,expiresAtMs:binding.expiresAtMs};
  const error=await this.ctx.blockConcurrencyWhile(async()=>{
   if(this.state.revoked)return 'device_revoked';
   if(this.state.binding)return JSON.stringify(this.state.binding)===JSON.stringify(row)?null:'device_binding_conflict';
   const state={binding:row,revoked:false};await this.ctx.storage.put('device',state);this.state=state;return null;
  });if(error)throw new Error(error);
 }
 async lookup(digest:string):Promise<DeviceBinding|null>{const row=this.state.binding;return !this.state.revoked&&row&&row.deviceDigest===digest&&row.expiresAtMs>Date.now()?structuredClone(row):null;}
 async consume(digest:string,proof:VerifiedDeviceProof):Promise<DeviceBinding|null>{
  if(!proof||typeof proof.jti!=='string'||!/^[-A-Za-z0-9_]{1,128}$/.test(proof.jti)||!Number.isSafeInteger(proof.issuedAt))return null;
  return this.ctx.blockConcurrencyWhile(async()=>{
   const row=this.state.binding;const now=Date.now();const expires=proof.issuedAt*1000+65000;
   if(!row||this.state.revoked||row.deviceDigest!==digest||row.deviceJkt!==proof.jkt||row.expiresAtMs<=now||expires<=now||proof.issuedAt*1000>now+5000)return null;
   const entries=await this.ctx.storage.list<number>({prefix:'proof:',limit:1025});
   for(const [key,expiry]of entries)if(expiry<=now){await this.ctx.storage.delete(key);entries.delete(key);}
   const key='proof:'+proof.jti;if(entries.has(key)||entries.size>=1024)return null;
   await this.ctx.storage.put(key,expires);return structuredClone(row);
  });
 }
 async revoke(owner:string):Promise<void>{
  const error=await this.ctx.blockConcurrencyWhile(async()=>{if(this.state.binding?.ownerId!==owner)return 'owner_mismatch';const state={...this.state,revoked:true};await this.ctx.storage.put('device',state);this.state=state;return null;});if(error)throw new Error(error);
 }
 async fetch(_request:Request):Promise<Response>{return new Response('not_found',{status:404});}
}
