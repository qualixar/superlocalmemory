import type {AuthRequest} from '@cloudflare/workers-oauth-provider';
export const AUTH_ISSUER='https://auth.superlocalmemory.com';
export const MCP_RESOURCE='https://mcp.superlocalmemory.com/mcp';
export const OWNER_RESOURCE=AUTH_ISSUER+'/owner';
const MEMORY_SCOPES=new Set(['slm:read','slm:write','slm:session']);
/** Extra pilot checks after the maintained provider has validated client and callback. */
export function validateAuthorizationRequest(request:AuthRequest):boolean {
 if(!request||request.issuer!==AUTH_ISSUER||request.responseType!=='code'||request.codeChallengeMethod!=='S256'||typeof request.codeChallenge!=='string'||!/^[-A-Za-z0-9_]{43}$/.test(request.codeChallenge)||typeof request.state!=='string'||!request.state||request.state.length>1024||!Array.isArray(request.scope)||new Set(request.scope).size!==request.scope.length)return false;
 if(request.resource===OWNER_RESOURCE)return request.scope.length===1&&request.scope[0]==='slm:connect';
 return request.resource===MCP_RESOURCE&&request.scope.includes('slm:read')&&request.scope.every(s=>MEMORY_SCOPES.has(s));
}
export function selectedScopes(requested:readonly string[],ceiling:{read:boolean;write:boolean;session:boolean}):string[]{
 if(!ceiling.read||!requested.includes('slm:read'))return [];
 return requested.filter(s=>s==='slm:read'||s==='slm:write'&&ceiling.write||s==='slm:session'&&ceiling.session);
}
/** Fetch exactly GitHub's identity endpoint: never trust caller names or token payloads. */
export async function verifyGithubIdentity(accessToken:string,fetcher:typeof fetch=fetch):Promise<string>{
 if(typeof accessToken!=='string'||!accessToken||accessToken.length>4096||/[\r\n]/.test(accessToken))throw new Error('identity_unavailable');
 const response=await fetcher('https://api.github.com/user',{headers:{Authorization:'Bearer '+accessToken,Accept:'application/vnd.github+json','User-Agent':'SuperLocalMemory','X-GitHub-Api-Version':'2022-11-28'},redirect:'error',signal:AbortSignal.timeout(10000)});
 if(!response.ok)throw new Error('identity_unavailable');
 const raw=await response.text();if(raw.length>65536)throw new Error('identity_unavailable');
 let value:unknown;try{value=JSON.parse(raw);}catch{throw new Error('identity_unavailable');}
 if(!value||typeof value!=='object'||!('id' in value)||typeof value.id!=='number'||!Number.isSafeInteger(value.id)||value.id<=0)throw new Error('identity_denied');
 return String(value.id);
}
