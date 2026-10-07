import type {OAuthHelpers,TokenSummary} from '@cloudflare/workers-oauth-provider';
import type {TokenIndexDO,IndexedToken} from './token-index-do.ts';
import type {IssuedTokenReference} from './contracts.ts';
export interface IssuedTokenEnv {TOKEN_INDEX:DurableObjectNamespace<TokenIndexDO>;}
export async function tokenDigest(token:string):Promise<string>{return [...new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(token)))].map(x=>x.toString(16).padStart(2,'0')).join('');}
export async function indexedToken(token:string,env:IssuedTokenEnv):Promise<IndexedToken|null>{
 if(!token||token.length>8192)return null;const digest=await tokenDigest(token);return env.TOKEN_INDEX.getByName(digest.slice(0,4)).lookup(digest);
}
/** The SDK decodes access tokens; refresh tokens are only opaque digests here. */
export async function recordIssuedTokens(body:{access_token?:unknown;refresh_token?:unknown},env:IssuedTokenEnv,api:Pick<OAuthHelpers,'unwrapToken'>,refreshValidUntil?:string):Promise<TokenSummary>{
 if(typeof body.access_token!=='string'||!body.access_token||body.access_token.length>8192)throw new Error('token_record_unavailable');
 const summary=await api.unwrapToken(body.access_token);const props=summary?.grant.props;
 if(!summary||!props||typeof props!=='object'||typeof props.ownerId!=='string'||summary.userId!==props.ownerId||typeof props.connectionId!=='string'||!Number.isSafeInteger(summary.expiresAt)||summary.expiresAt<=Date.now()/1000)throw new Error('token_record_unavailable');
 const audience=typeof summary.audience==='string'?summary.audience:null;
 if(audience!=='https://mcp.superlocalmemory.com/mcp'&&audience!=='https://auth.superlocalmemory.com/owner')throw new Error('token_record_unavailable');
 const authorizationId=props.kind==='native'?'native:'+props.connectionId:props.authorizationId;
 if(typeof authorizationId!=='string'||!summary.grant.clientId)throw new Error('token_record_unavailable');
 const binding={authorizationId,ownerId:props.ownerId,connectionId:props.connectionId,clientId:summary.grant.clientId,audience};
 const refs:IssuedTokenReference[]=[{...binding,tokenDigest:await tokenDigest(body.access_token),tokenKind:'access',validUntil:new Date(summary.expiresAt*1000).toISOString()}];
 if(body.refresh_token!==undefined){
  if(typeof body.refresh_token!=='string'||!body.refresh_token||body.refresh_token.length>8192)throw new Error('token_record_unavailable');
  const validUntil=refreshValidUntil??new Date(Date.now()+30*24*3600*1000).toISOString();
  if(!Number.isFinite(Date.parse(validUntil))||Date.parse(validUntil)<=Date.now())throw new Error('token_record_unavailable');
  refs.push({...binding,tokenDigest:await tokenDigest(body.refresh_token),tokenKind:'refresh',validUntil});
 }
 for(const ref of refs)await env.TOKEN_INDEX.getByName(ref.tokenDigest.slice(0,4)).record([ref]);
 return summary;
}
