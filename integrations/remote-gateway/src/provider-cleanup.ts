/** Private metadata-only retry ledger; never stores access or refresh tokens. */
interface CleanupEnv {OAUTH_KV:KVNamespace;}
interface CleanupApi {revokeGrant(grantId:string,userId:string):Promise<void>;}
interface Reference {userId:string;grantId:string;}
function valid(value:unknown):value is Reference{return !!value&&typeof value==='object'&&['userId','grantId'].every(key=>{const part=(value as Record<string,unknown>)[key];return typeof part==='string'&&part.length>0&&part.length<=256;});}
export async function queueProviderCleanup(env:CleanupEnv,reference:Reference):Promise<void>{
 if(!valid(reference))throw new Error('invalid_cleanup_reference');
 await env.OAUTH_KV.put('slm-cleanup:'+crypto.randomUUID(),JSON.stringify({userId:reference.userId,grantId:reference.grantId}),{expirationTtl:30*24*3600});
}
export async function retryProviderCleanup(env:CleanupEnv,api:CleanupApi):Promise<void>{
 try{
  const page=await env.OAUTH_KV.list({prefix:'slm-cleanup:',limit:5});
  for(const key of page.keys){try{
   const raw=await env.OAUTH_KV.get(key.name);if(raw===null)continue;if(raw.length>2048)throw new Error('invalid_cleanup_reference');
   const reference:unknown=JSON.parse(raw);if(!valid(reference))throw new Error('invalid_cleanup_reference');
   await api.revokeGrant(reference.grantId,reference.userId);await env.OAUTH_KV.delete(key.name);
  }catch{console.warn('provider_cleanup_retry_pending');}}
 }catch{console.warn('provider_cleanup_retry_unavailable');}
}
