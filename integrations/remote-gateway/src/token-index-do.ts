import {DurableObject} from 'cloudflare:workers';
import type {IssuedTokenReference} from './contracts.ts';
export interface IndexedToken extends IssuedTokenReference {revoked:boolean;}
const digest=(value:unknown):value is string=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
const id=(value:unknown):value is string=>typeof value==='string'&&/^[A-Za-z0-9_.:-]{1,256}$/.test(value);
const client=(value:unknown):value is string=>typeof value==='string'&&value.length>0&&value.length<=2048&&value.trim()===value&&!/[\x00-\x1f\x7f]/.test(value);
function valid(ref:IssuedTokenReference):boolean {
  if(!ref||!digest(ref.tokenDigest)||!['access','refresh'].includes(ref.tokenKind)||![ref.ownerId,ref.authorizationId,ref.connectionId].every(id)||!client(ref.clientId))return false;
  if(!['https://mcp.superlocalmemory.com/mcp','https://auth.superlocalmemory.com/owner'].includes(ref.audience)||typeof ref.validUntil!=='string'||ref.validUntil.length>32)return false;
  const expiry=Date.parse(ref.validUntil);return Number.isFinite(expiry)&&expiry>Date.now()&&new Date(expiry).toISOString()===ref.validUntil;
}
/** Private issuer ledger: only token digests and validated binding metadata.
 * Separate prefix buckets bound registry size. No raw access/refresh tokens.
 */
export class TokenIndexDO extends DurableObject<Record<string,unknown>> {
  async record(refs:IssuedTokenReference[]):Promise<void>{
    if(!Array.isArray(refs)||!refs.length||refs.length>2||!refs.every(valid))throw new Error('invalid_token_reference');
    const curated=refs.map(ref=>({tokenDigest:ref.tokenDigest,tokenKind:ref.tokenKind,authorizationId:ref.authorizationId,ownerId:ref.ownerId,clientId:ref.clientId,connectionId:ref.connectionId,audience:ref.audience,validUntil:ref.validUntil,revoked:false} satisfies IndexedToken));
    const error=await this.ctx.blockConcurrencyWhile(async()=>{
      const rows=await this.ctx.storage.list<IndexedToken>({prefix:'token:',limit:2001});
      for(const [key,row] of rows)if(Date.parse(row.validUntil)<=Date.now()){await this.ctx.storage.delete(key);rows.delete(key);}
      const additions=curated.filter(ref=>!rows.has('token:'+ref.tokenDigest));
      if(rows.size+additions.length>2000)return 'token_capacity_exhausted';
      for(const ref of curated){const old=rows.get('token:'+ref.tokenDigest);if(old?.revoked)return 'token_revoked';if(old&&JSON.stringify(old)!==JSON.stringify(ref))return 'token_conflict';}
      await this.ctx.storage.put(Object.fromEntries(curated.map(ref=>['token:'+ref.tokenDigest,ref])));return null;
    });
    if(error)throw new Error(error);
  }
  async lookup(value:string):Promise<IndexedToken|null>{
    if(!digest(value))return null;
    const row=await this.ctx.storage.get<IndexedToken>('token:'+value);
    return row&&Date.parse(row.validUntil)>Date.now()?row:null;
  }
  async revoke(value:string,clientId:string):Promise<boolean>{
    if(!digest(value)||!client(clientId))return false;
    return this.ctx.blockConcurrencyWhile(async()=>{
      const key='token:'+value;const row=await this.ctx.storage.get<IndexedToken>(key);
      if(!row||row.clientId!==clientId)return false;
      if(!row.revoked)await this.ctx.storage.put(key,{...row,revoked:true});
      return true;
    });
  }
  async fetch(_request:Request):Promise<Response>{return new Response('not_found',{status:404});}
}
