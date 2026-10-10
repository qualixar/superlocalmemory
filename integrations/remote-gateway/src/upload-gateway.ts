import type {RelayDO} from './relay-do.ts';
import type {RegistryDO} from './registry-do.ts';
import {RELAY_DEADLINE_MS} from './relay-protocol.ts';
import {publicRelayCode} from './relay-errors.ts';
import {CHUNK_BYTES,HARD_MAX_BYTES,UPLOAD_HEADER,UploadAbort,chunksOf,cleanReply,limitKey,newNonce,parseUploadPath,relayProblem,statusFor,toBase64,uploadHeader,type UploadReply} from './upload-protocol.ts';
import {jsonReply,messagePage,uploadPage} from './upload-page.ts';

/** The public one-time upload link: `GET /u/<connection>/<token>` serves a picker, `POST` streams the file to that connection's
 * laptop as relay request frames. The gateway keeps no copy of the file and never decides whether the token is good: the laptop does. */
export interface UploadEnv {RELAYS:DurableObjectNamespace<RelayDO>;REGISTRIES:DurableObjectNamespace<RegistryDO>;UPLOAD_IP?:RateLimit;UPLOAD_CONN?:RateLimit;}
export const SELF_ORIGIN='https://mcp.superlocalmemory.com';
/** One finish frame waits up to 15 s on the laptop; asking this many times covers a slow first-time picture model. */
export const FINISH_TRIES=8;
const STILL_SAVING='Your computer is still saving this file. You can close this page; it will appear in your memory shortly.';

type Hop={kind:'reply';reply:UploadReply}|{kind:'problem';status:number;message:string};
type Link={connection:string;token:string;nonce:string};
const REVOKED='This upload link no longer works. Ask the app for a new one.';

async function hop(env:UploadEnv,link:Link,op:'info'|'chunk'|'finish',index:number,total:number,body:Uint8Array=new Uint8Array()):Promise<Hop> {
  let response:Response;
  try {
    response=await env.RELAYS.getByName(link.connection).forwardCurrent({v:1,kind:'request',id:crypto.randomUUID(),deadlineAt:Date.now()+RELAY_DEADLINE_MS,
      headers:[['content-type','application/octet-stream'],[UPLOAD_HEADER,uploadHeader(op,link.token,index,total,link.nonce)]],bodyBase64:toBase64(body)});
  } catch { return {kind:'problem',...relayProblem('origin_unavailable')}; }
  if(response.status!==200){
    let code='origin_unavailable';
    try{code=publicRelayCode(await response.json());}catch{/* a body that is not ours is withheld */}
    return {kind:'problem',...relayProblem(code)};
  }
  try{return {kind:'reply',reply:cleanReply(await response.json())};}
  catch{return {kind:'reply',reply:cleanReply(null)};}
}

/** The app's consent for this connection is still on record (not revoked, access not ended, saving pictures allowed). Anything unclear counts as no. */
async function consentActive(env:UploadEnv,connection:string):Promise<boolean> {
  try{return (await env.REGISTRIES.getByName(connection).uploadsAllowed())===true;}catch{return false;}
}

/** Per address and per link (connection plus a hash of the token); a missing or broken limiter refuses rather than lets everything through. */
async function admission(request:Request,env:UploadEnv,link:Link):Promise<{status:number;code:string;message:string}|null> {
  if(!env.UPLOAD_IP||!env.UPLOAD_CONN)return {status:503,code:'upload_unavailable',message:'Uploads are not available right now. Try again in a minute.'};
  try {
    const address=request.headers.get('CF-Connecting-IP')??'unknown';
    const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(address));
    const key=Array.from(new Uint8Array(digest),value=>value.toString(16).padStart(2,'0')).join('');
    if(!(await env.UPLOAD_IP.limit({key})).success||!(await env.UPLOAD_CONN.limit({key:await limitKey(link.connection,link.token)})).success)return {status:429,code:'rate_limited',message:'Too many tries. Wait a minute and try again.'};
    return null;
  } catch { return {status:503,code:'upload_unavailable',message:'Uploads are not available right now. Try again in a minute.'}; }
}

function refusal(page:boolean,status:number,code:string,message:string):Response {
  const response=page?messagePage(status,message):jsonReply(status,{ok:false,code,message});
  if(status===429)response.headers.set('Retry-After','60');
  return response;
}

async function showPicker(env:UploadEnv,link:Link):Promise<Response> {
  if(!(await consentActive(env,link.connection)))return messagePage(410,REVOKED);
  const answer=await hop(env,link,'info',0,0);
  if(answer.kind==='problem')return messagePage(answer.status,answer.message);
  const {reply}=answer;
  if(!reply.ok)return messagePage(statusFor(reply.code??'error'),reply.message??'This upload link cannot be used.');
  if(!reply.kind||reply.maxBytes===undefined)return messagePage(502,'Your computer sent an answer this page could not use.');
  return uploadPage(reply.kind,reply.maxBytes);
}

function declaredLength(request:Request):{length:number}|{status:number;code:string;message:string} {
  const header=request.headers.get('content-length');
  if(header===null)return {status:411,code:'length_required',message:'The file size was not sent. Try again.'};
  if(!/^\d{1,12}$/.test(header))return {status:400,code:'bad_length',message:'The file size was not understood.'};
  const length=Number(header);
  if(length===0)return {status:400,code:'empty',message:'That file is empty.'};
  if(length>HARD_MAX_BYTES)return {status:413,code:'too_large',message:'That file is too large.'};
  return {length};
}

/** Sends every chunk in order; the first problem or refusal ends it. */
async function sendChunks(env:UploadEnv,link:Link,body:ReadableStream<Uint8Array>,declared:number):Promise<Response|null> {
  let index=0,sent=0;
  try {
    for await(const piece of chunksOf(body,CHUNK_BYTES,declared)){
      const answer=await hop(env,link,'chunk',index,declared,piece);
      if(answer.kind==='problem')return refusal(false,answer.status,'unreachable',answer.message);
      if(!answer.reply.ok)return refusal(false,statusFor(answer.reply.code??'error'),answer.reply.code??'error',answer.reply.message??'The file could not be saved.');
      index++;sent+=piece.length;
    }
  } catch(error) {
    if(error instanceof UploadAbort)return refusal(false,400,error.code,'The file changed size while it was being sent. Try again.');
    return refusal(false,502,'interrupted','The upload was interrupted. Try again.');
  }
  return sent===declared?null:refusal(false,400,'size_mismatch','The file changed size while it was being sent. Try again.');
}

async function finish(env:UploadEnv,link:Link,declared:number):Promise<Response> {
  if(!(await consentActive(env,link.connection)))return refusal(false,410,'revoked',REVOKED);
  for(let attempt=0;attempt<FINISH_TRIES;attempt++){
    const answer=await hop(env,link,'finish',0,declared);
    if(answer.kind==='problem')return refusal(false,answer.status,'unreachable',answer.message);
    const {reply}=answer;
    if(!reply.ok)return refusal(false,statusFor(reply.code??'error'),reply.code??'error',reply.message??'The file could not be saved.');
    if(reply.done===true)return jsonReply(200,{ok:true,done:true,message:reply.message??'Saved to your memory.'});
  }
  return jsonReply(202,{ok:true,done:false,message:STILL_SAVING});
}

async function receive(request:Request,env:UploadEnv,link:Link):Promise<Response> {
  const origin=request.headers.get('Origin');
  if(origin!==null&&origin!==SELF_ORIGIN)return refusal(false,403,'origin_denied','This upload must be started from the link page.');
  const length=declaredLength(request);
  if('status' in length)return refusal(false,length.status,length.code,length.message);
  if(request.body===null)return refusal(false,400,'empty','That file is empty.');
  if(!(await consentActive(env,link.connection)))return refusal(false,410,'revoked',REVOKED);
  const info=await hop(env,link,'info',0,0);
  if(info.kind==='problem')return refusal(false,info.status,'unreachable',info.message);
  if(!info.reply.ok)return refusal(false,statusFor(info.reply.code??'error'),info.reply.code??'error',info.reply.message??'This upload link cannot be used.');
  const limit=info.reply.maxBytes;
  if(limit===undefined)return refusal(false,502,'error','Your computer sent an answer this page could not use.');
  if(length.length>limit)return refusal(false,413,'too_large',`That file is too large. The limit is ${Math.max(1,Math.floor(limit/(1024*1024)))} MB.`);
  const failed=await sendChunks(env,link,request.body,length.length);
  return failed??finish(env,link,length.length);
}

/** `null` when the request is not for an upload link, so the caller carries on with its own routes. */
export async function handleUpload(request:Request,env:UploadEnv):Promise<Response|null> {
  const path=new URL(request.url).pathname;
  if(!path.startsWith('/u/'))return null;
  const page=request.method==='GET';
  const parsed=parseUploadPath(path);
  if(!parsed)return refusal(true,404,'invalid_link','This upload link is not valid.');
  const link:Link={...parsed,nonce:newNonce()};
  if(request.method!=='GET'&&request.method!=='POST'){
    const response=refusal(true,405,'method_not_allowed','This address only takes a file picked on its page.');
    response.headers.set('Allow','GET, POST');
    return response;
  }
  const blocked=await admission(request,env,link);
  if(blocked)return refusal(page,blocked.status,blocked.code,blocked.message);
  return page?showPicker(env,link):receive(request,env,link);
}
