import {OAuthAuthorizationServer,OAuthError,type TokenExchangeCallbackOptions} from '@cloudflare/workers-oauth-provider';
import {GRANT_IDLE_TTL_S} from './credential-lifetime.ts';
import {AUTH_ISSUER,MCP_RESOURCE,OWNER_RESOURCE} from './authorization-policy.ts';
import type {RegistryDO} from './registry-do.ts';
import type {OwnerIndexDO} from './owner-index-do.ts';
import type {BootstrapDO} from './bootstrap-do.ts';
import type {AuthProps,Scope,VerifiedActor} from './contracts.ts';
export interface NativeAuthProps {kind:'native';ownerId:string;installationId:string;profileId:string;connectionId:string;deviceJkt:string;}
export interface AuthorizationEnv {
 OAUTH_KV:KVNamespace;
 REGISTRIES:DurableObjectNamespace<RegistryDO>;
 OWNERS:DurableObjectNamespace<OwnerIndexDO>;
 BOOTSTRAPS:DurableObjectNamespace<BootstrapDO>;
}
const record=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const owner=(value:unknown):value is string=>typeof value==='string'&&/^[0-9]{1,32}$/.test(value);
const identifier=(value:unknown):value is string=>typeof value==='string'&&/^[-A-Za-z0-9_.:]{1,256}$/.test(value);
function denied():never {throw new OAuthError('invalid_grant',{description:'Authorization is unavailable'});}
/** Called after the provider validates code/PKCE or refresh credentials. Rechecks
 * the authoritative local connection ceiling on every token issuance. */
export async function currentAuthorization(options:TokenExchangeCallbackOptions<AuthorizationEnv>):Promise<void>{
 const props:unknown=options.props;
 if(!record(props)||!owner(props.ownerId)||options.userId!==props.ownerId||options.clientId!==options.subjectClientId||!identifier(props.connectionId))denied();
 if(options.resource===OWNER_RESOURCE){
  if(props.kind!=='native'||!identifier(props.installationId)||!identifier(props.profileId)||typeof props.deviceJkt!=='string'||!options.requestedScope.every(s=>s==='slm:connect')||!options.requestedScope.includes('slm:connect'))denied();
  const available=await options.env.OWNERS.getByName(props.ownerId).authorizeNative(props.ownerId,props.installationId,props.profileId,options.clientId,props.deviceJkt);
  if(!available)denied();
  if(options.grantType==='authorization_code'){
   const bootstrap=await options.env.BOOTSTRAPS.getByName(props.connectionId).get();
   if(!bootstrap||bootstrap.status!=='completed'||bootstrap.ownerId!==props.ownerId||bootstrap.authRequest.clientId!==options.clientId)denied();
  }else{
   const connection=await options.env.OWNERS.getByName(props.ownerId).getConnection(props.ownerId,props.connectionId);
   if(!connection||connection.revokedAt!==null||connection.installationId!==props.installationId||connection.profileId!==props.profileId||connection.deviceJkt!==props.deviceJkt)denied();
  }
  return;
 }
 if(options.resource!==MCP_RESOURCE||!identifier(props.authorizationId)||props.kind==='native')denied();
 const actor:VerifiedActor={ownerId:props.ownerId,connectionId:props.connectionId,authorizationId:props.authorizationId,clientId:options.clientId,audience:MCP_RESOURCE,credentialKind:'oauth',scopes:options.requestedScope as Scope[]};
 const admission=await options.env.REGISTRIES.getByName(props.connectionId).admit(actor,MCP_RESOURCE,{era:'legacy',rpcMethod:'initialize',rpcId:1,originalBody:new Uint8Array()});
 if(!admission.allowed)denied();
}
/** Protocol-owned endpoints only. Interactive application routes are composed
 * in the auth Worker; this library owns DCR, PKCE, code/token handling and metadata. */
export const authorizationServer=new OAuthAuthorizationServer<AuthorizationEnv>({
 issuer:AUTH_ISSUER,resources:[MCP_RESOURCE,OWNER_RESOURCE],
 // Hosted connector flows that cannot send RFC 8707 resource get the MCP audience.
 // Native enrollment still requires the explicit owner resource and slm:connect.
 defaultResource:MCP_RESOURCE,
 authorizeEndpoint:AUTH_ISSUER+'/authorize',tokenEndpoint:AUTH_ISSUER+'/oauth/token',
 clientRegistrationEndpoint:AUTH_ISSUER+'/oauth/register',
 scopesSupported:['slm:read','slm:write','slm:session','slm:mesh','slm:media','slm:connect'],
 accessTokenTTL:3600,refreshTokenTTL:GRANT_IDLE_TTL_S,
 // A grant in use slides forward on every refresh; one left unused for 30 days expires.
 refreshTokenIdleTTL:GRANT_IDLE_TTL_S,
 clientIdMetadataDocumentEnabled:true,
 tokenExchangeCallback:async options=>{await currentAuthorization(options);},
});
export type MemoryAuthProps=AuthProps;
