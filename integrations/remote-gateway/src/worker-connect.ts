import {verifyDeviceProof} from './device-proof.ts';
import {tokenDigest} from './issued-token-protocol.ts';
import type {DeviceIndexDO} from './device-index-do.ts';
import type {OwnerIndexDO} from './owner-index-do.ts';
import type {RelayDO} from './relay-do.ts';
export interface ConnectEnv {DEVICES:DurableObjectNamespace<DeviceIndexDO>;OWNERS:DurableObjectNamespace<OwnerIndexDO>;RELAYS:DurableObjectNamespace<RelayDO>;}
const ENDPOINT='https://connect.superlocalmemory.com/connector';
function denial(status:number,error:string):Response{return Response.json({error},{status,headers:{'Cache-Control':'no-store'}});}
export async function connectFetch(request:Request,env:ConnectEnv,_ctx:ExecutionContext):Promise<Response>{
 const url=new URL(request.url);if(url.origin!==new URL(ENDPOINT).origin||url.pathname!=='/connector'||url.search)return denial(403,'host_denied');
 if(request.method!=='GET')return denial(405,'method_not_allowed');
 if(request.headers.get('Origin')!==null)return denial(403,'origin_denied');
 if(request.headers.get('Upgrade')?.toLowerCase()!=='websocket')return denial(426,'upgrade_required');
 const token=/^Bearer ([A-Za-z0-9_-]{32,256})$/.exec(request.headers.get('Authorization')??'')?.[1];const proof=request.headers.get('DPoP');
 if(!token||!proof)return denial(401,'device_unauthorized');
 try{
  const digest=await tokenDigest(token);const device=env.DEVICES.getByName(digest);const binding=await device.lookup(digest);if(!binding)return denial(401,'device_unauthorized');
  let verified;try{verified=await verifyDeviceProof(proof,{jkt:binding.deviceJkt,method:'GET',url:ENDPOINT,token});}catch{return denial(401,'device_unauthorized');}
  const consumed=await device.consume(digest,verified);if(!consumed)return denial(401,'device_unauthorized');
  const current=await env.OWNERS.getByName(binding.ownerId).getConnection(binding.ownerId,binding.connectionId);
  if(!current||current.revokedAt!==null||current.deviceDigest!==digest||current.deviceJkt!==binding.deviceJkt||current.installationId!==binding.installationId||current.profileId!==binding.profileId||current.deviceExpiresAtMs<=Date.now())return denial(403,'connection_unavailable');
  return await env.RELAYS.getByName(binding.connectionId).fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+token,...(request.headers.get('x-slm-connector-features')==='grant-v1'?{'x-slm-connector-features':'grant-v1'}:{})}}));
 }catch{return denial(503,'connector_unavailable');}
}
export default {fetch:connectFetch};
