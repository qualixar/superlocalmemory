import {OAuthResourceServer,type OAuthResourceContext,type OAuthResourceTokenValidation} from '@cloudflare/workers-oauth-provider';
import type {AuthProps,Scope,VerifiedActor} from './contracts.ts';
import {RegistryDO} from './registry-do.ts';
import {RelayDO} from './relay-do.ts';
import {RELAY_DEADLINE_MS} from './relay-protocol.ts';
import {publicRelayCode} from './relay-errors.ts';
import {secondsUntilUtcMidnight} from './usage-limit.ts';
import {GatewayInputError,filterMcpResponse,parseMcpRequest} from './mcp-http.ts';

export {RegistryDO,RelayDO};
export const RESOURCE='https://mcp.superlocalmemory.com/mcp';
export const ISSUER='https://auth.superlocalmemory.com';
export interface ResourceEnv {
  AUTH_SERVER:{validateToken(resource:string,token:string):Promise<OAuthResourceTokenValidation<AuthProps>|null>};
  REGISTRIES:DurableObjectNamespace<RegistryDO>;
  RELAYS:DurableObjectNamespace<RelayDO>;
}
function failure(status:number,error:string):Response {return Response.json({error},{status,headers:{'Cache-Control':'no-store'}});}
function actor(context:OAuthResourceContext<AuthProps>):VerifiedActor|null {
  const props=context.props;const auth=context.auth;
  if(!props||!auth||auth.userId!==props.ownerId||!auth.clientId||!Number.isSafeInteger(auth.expiresAt)||auth.expiresAt!*1000<=Date.now())return null;
  return {ownerId:props.ownerId,authorizationId:props.authorizationId,connectionId:props.connectionId,
    clientId:auth.clientId,audience:auth.audience,credentialKind:'oauth',scopes:auth.scope as Scope[]};
}
function base64(bytes:Uint8Array):string {
  const pieces:string[]=[];for(let i=0;i<bytes.length;i+=16384)pieces.push(String.fromCharCode(...bytes.subarray(i,i+16384)));
  return btoa(pieces.join(''));
}
async function handle(request:Request,env:ResourceEnv,context:OAuthResourceContext<AuthProps>):Promise<Response> {
  const origin=request.headers.get('Origin');
  if(origin!==null&&origin!==new URL(RESOURCE).origin&&origin!==ISSUER)return failure(403,'origin_denied');
  const principal=actor(context);
  if(!principal)return failure(401,'invalid_principal');
  if(!principal.scopes.includes('slm:read'))return failure(403,'insufficient_scope');
  try {
    const parsed=await parseMcpRequest(request);
    const registry=env.REGISTRIES.getByName(principal.connectionId);
    const admission=await registry.admit(principal,RESOURCE,parsed.envelope);
    if(!admission.allowed){
      const refused=failure(admission.httpStatus,admission.code);
      // Tells the client when the daily tool-call quota resets.
      if(admission.code==='DAILY_LIMIT_REACHED')refused.headers.set('Retry-After',String(secondsUntilUtcMidnight(Date.now())));
      return refused;
    }
    if(admission.grant.connection.origin.kind!=='relay')return failure(503,'origin_transport_unavailable');
    const relay=env.RELAYS.getByName(admission.grant.connection.connectionId);
    const identifier=crypto.randomUUID();
    const abort=()=>{context.waitUntil(relay.cancelCaller(identifier).catch(()=>{console.warn('relay_cancel_unavailable');}));};
    request.signal.addEventListener('abort',abort,{once:true});
    let response:Response;
    try {
      if(request.signal.aborted)return failure(499,'request_cancelled');
      response=await relay.forwardCurrent({v:1,kind:'request',id:identifier,deadlineAt:Date.now()+RELAY_DEADLINE_MS,
        headers:parsed.headers,bodyBase64:base64(parsed.envelope.originalBody)},
        {requestParamHeaders:parsed.parameterHeaderNames});
    }finally{request.signal.removeEventListener('abort',abort);}
    if(response.status>=400){
      // Relay failure packets contain bounded non-secret codes; never expose
      // arbitrary origin errors or HTML to the public client.
      let code='origin_unavailable';
      try{code=publicRelayCode(await response.json());}catch{}
      return failure(response.status,code);
    }
    const headers=new Headers({'Content-Type':'application/json','Cache-Control':'no-store'});
    const protocol=response.headers.get('mcp-protocol-version');if(protocol)headers.set('mcp-protocol-version',protocol);
    if(parsed.envelope.rpcId===undefined)return new Response(null,{status:response.status,headers});
    const body=filterMcpResponse(new Uint8Array(await response.arrayBuffer()),parsed.envelope,admission.grant.allowedTools);
    return new Response(body,{status:response.status,headers});
  }catch(error){
    if(error instanceof GatewayInputError)return Response.json({jsonrpc:'2.0',id:error.rpcId,error:{code:error.rpcCode,message:error.code,...(error.data!==undefined?{data:error.data}:{})}},{status:error.status,headers:{'Cache-Control':'no-store'}});
    return failure(503,'gateway_unavailable');
  }
}
const protectedResource=new OAuthResourceServer<ResourceEnv,AuthProps>({
  resourceMetadata:{resource:RESOURCE,authorization_servers:[ISSUER]},requiredScopes:['slm:read'],
  validateToken:env=>(resource,token)=>env.AUTH_SERVER.validateToken(resource,token),handler:{fetch:handle},
});
export const resourceGateway={
  async fetch(request:Request,env:ResourceEnv,context:ExecutionContext):Promise<Response>{
    if(new URL(request.url).origin!==new URL(RESOURCE).origin)return failure(403,'host_denied');
    return protectedResource.fetch(request,env,context);
  },
};
export default resourceGateway;
