import {calculateJwkThumbprint,decodeProtectedHeader,importJWK,jwtVerify,type JWK} from 'jose';
export class DeviceProofError extends Error {constructor(){super('invalid_device_proof');}}
export interface ProofExpectation {jkt:string;method:string;url:string;token?:string;}
export interface VerifiedDeviceProof {jti:string;issuedAt:number;jkt:string;}
export async function tokenHash(token:string):Promise<string>{
  const bytes=new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(token)));
  return btoa(String.fromCharCode(...bytes)).replaceAll('+','-').replaceAll('/','_').replace(/=+$/,'');
}
export function publicDeviceKey(value:unknown):JWK {
  if(!value||typeof value!=='object'||Array.isArray(value))throw new DeviceProofError();
  const key=value as Record<string,unknown>;
  if(key.kty!=='EC'||key.crv!=='P-256'||typeof key.x!=='string'||typeof key.y!=='string'||!/^[-A-Za-z0-9_]{43}$/.test(key.x)||!/^[-A-Za-z0-9_]{43}$/.test(key.y)||Object.hasOwn(key,'d'))throw new DeviceProofError();
  return {kty:'EC',crv:'P-256',x:key.x,y:key.y};
}
/** Private installation/control proof using RFC9449 JWT claims and maintained
 * JOSE verification. A separate durable authority consumes jti before access.
 * This function does not itself issue OAuth tokens or claim replay protection.
 */
export async function verifyDeviceProof(proof:string,expected:ProofExpectation):Promise<VerifiedDeviceProof>{
  try{
    if(typeof proof!=='string'||proof.length>4096||!/^[-A-Za-z0-9_]{43}$/.test(expected.jkt))throw new DeviceProofError();
    const header=decodeProtectedHeader(proof);
    if(header.typ!=='dpop+jwt'||header.alg!=='ES256'||Object.keys(header).some(key=>!['typ','alg','jwk'].includes(key)))throw new DeviceProofError();
    const jwk=publicDeviceKey(header.jwk);
    const jkt=await calculateJwkThumbprint(jwk);
    if(jkt!==expected.jkt)throw new DeviceProofError();
    const key=await importJWK(jwk,'ES256');
    const {payload}=await jwtVerify(proof,key,{algorithms:['ES256'],typ:'dpop+jwt',maxTokenAge:'60s',clockTolerance:5});
    const url=new URL(expected.url);url.search='';url.hash='';
    if(payload.htm!==expected.method||payload.htu!==url.href||!Number.isSafeInteger(payload.iat)||typeof payload.jti!=='string'||!/^[-A-Za-z0-9_]{1,128}$/.test(payload.jti))throw new DeviceProofError();
    if(expected.token!==undefined&&payload.ath!==await tokenHash(expected.token))throw new DeviceProofError();
    return {jti:payload.jti,issuedAt:payload.iat!,jkt};
  }catch{throw new DeviceProofError();}
}
