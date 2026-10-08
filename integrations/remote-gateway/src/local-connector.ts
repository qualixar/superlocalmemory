import { decodeRelayFrame, encodeRelayFrame, MAX_RESPONSE_BYTES, MAX_FRAME_BYTES, RELAY_DEADLINE_MS, CLOCK_SKEW_TOLERANCE_MS, type RelayFrame, type RelayCodecOptions } from './relay-protocol.ts';

interface ConnectorOptions {
  origin: string;
  originHeaders: Readonly<Record<string,string>>;
  fetcher?: typeof fetch;
  codecOptions?: RelayCodecOptions;
  send: (text:string) => void;
  close: (code:string) => void;
}
interface Operation { controller: AbortController; timer: ReturnType<typeof setTimeout>; }

function fromBase64(text:string):Uint8Array<ArrayBuffer> {
  const decoded=atob(text), bytes=new Uint8Array(decoded.length);
  for(let index=0;index<decoded.length;index++)bytes[index]=decoded.charCodeAt(index);
  return bytes;
}
function toBase64(bytes:Uint8Array):string {
  const parts:string[]=[];
  for(let index=0;index<bytes.length;index+=8192)
    parts.push(String.fromCharCode(...bytes.subarray(index,index+8192)));
  return btoa(parts.join(''));
}

/** Credentials never enter a URL or caller-controlled redirect. A private
 * origin adapter supplies only headers minted by the local SLM installation.
 * This transport is not the OAuth/grant policy authority.
 */
export function connectorSocketOptions(endpoint:string, deviceToken:string) {
  const url = new URL(endpoint);
  if(url.protocol!=='wss:' || url.hostname!=='connect.superlocalmemory.com' || url.port ||
      url.pathname!=='/connector' || url.search || url.hash || url.username || url.password ||
      !/^[A-Za-z0-9_-]{32,256}$/.test(deviceToken)) throw new Error('invalid_connector_configuration');
  return {url:url.href, options:{headers:{Authorization:`Bearer ${deviceToken}`},followRedirects:false,
    perMessageDeflate:false,maxPayload:MAX_FRAME_BYTES,handshakeTimeout:10000}};
}

async function boundedBody(response:Response,signal:AbortSignal):Promise<Uint8Array> {
  if(!response.body)return new Uint8Array();
  const reader=response.body.getReader(), parts:Uint8Array[]=[];
  const abort=()=>{void reader.cancel().catch(()=>{});};
  signal.addEventListener('abort',abort,{once:true});
  if(signal.aborted)abort();
  let size=0;
  try {
    for(;;){const next=await reader.read();if(next.done)break;
      size+=next.value.length;
      if(size>MAX_RESPONSE_BYTES){await reader.cancel();throw new Error('origin_response_too_large');}
      parts.push(next.value);
    }
  } finally {signal.removeEventListener('abort',abort);reader.releaseLock();}
  const bytes=new Uint8Array(size);let offset=0;
  for(const part of parts){bytes.set(part,offset);offset+=part.length;}
  return bytes;
}

/** One session per authenticated socket; stop fences every in-flight response.
 * Reconnect must construct a new instance and receive a new ready generation.
 */
export class LocalRelaySession {
  private readonly options:ConnectorOptions;
  private generation:number|null=null;
  private stopped=false;
  private operations=new Map<string,Operation>();

  get ready():boolean {return !this.stopped && this.generation!==null;}

  constructor(options:ConnectorOptions) {
    const url=new URL(options.origin);
    if(url.protocol!=='http:' || url.hostname!=='127.0.0.1' ||
       !['/mcp','/mcp/'].includes(url.pathname) || url.search || url.hash || url.username || url.password)
      throw new Error('invalid_local_origin');
    const headers=new Headers(options.originHeaders);
    for(const name of headers.keys())if(!['x-install-token','x-slm-api-key'].includes(name))throw new Error('invalid_origin_header');
    const codecOptions={requestParamHeaders:Object.freeze([...(options.codecOptions?.requestParamHeaders??[])])};
    if(!decodeRelayFrame('{"v":1,"kind":"cancel","id":"validation","generation":1}',codecOptions).ok)throw new Error('invalid_local_schema');
    this.options={...options,origin:url.href,originHeaders:Object.freeze({...options.originHeaders}),codecOptions:Object.freeze(codecOptions)};
  }

  stop():void {
    this.stopped=true;
    for(const operation of this.operations.values()){clearTimeout(operation.timer);operation.controller.abort();}
    this.operations.clear();
  }

  async receive(text:string):Promise<void> {
    if(this.stopped)return;
    if(this.generation===null){
      let ready:unknown;try{ready=JSON.parse(text);}catch{this.fail();return;}
      const r=ready as Record<string,unknown>|null;
      if(!r || typeof r!=='object' || Array.isArray(r) || Object.keys(r).length!==3 ||
        r.v!==1 || r.kind!=='ready' || !Number.isSafeInteger(r.generation) || (r.generation as number)<1 || JSON.stringify(r)!==text){this.fail();return;}
      this.generation=r.generation as number;return;
    }
    const decoded=decodeRelayFrame(text,this.options.codecOptions);
    if(!decoded.ok || decoded.frame.generation!==this.generation || decoded.frame.kind==='response'){this.fail();return;}
    const frame=decoded.frame;
    if(frame.kind==='cancel'){
      const operation=this.operations.get(frame.id);
      if(operation){clearTimeout(operation.timer);operation.controller.abort();this.operations.delete(frame.id);}
      return;
    }
    const remaining=frame.deadlineAt-Date.now();
    if(remaining<=0){this.reply(frame,504,'relay_timeout');return;}
    if(remaining>RELAY_DEADLINE_MS+CLOCK_SKEW_TOLERANCE_MS){this.reply(frame,400,'invalid_deadline');return;}
    const duration=Math.min(remaining,RELAY_DEADLINE_MS);
    if(this.operations.has(frame.id)){this.fail();return;}
    if(this.operations.size>=8){this.reply(frame,429,'connector_busy');return;}
    const controller=new AbortController();
    const operation={controller,timer:setTimeout(()=>controller.abort(),duration)};
    this.operations.set(frame.id,operation);
    await this.execute(frame,operation);
  }

  private async execute(frame:Extract<RelayFrame,{kind:'request'}>,operation:Operation):Promise<void> {
    try {
      const headers=new Headers(frame.headers.map(pair=>[pair[0],pair[1]]));
      for(const [name,value] of Object.entries(this.options.originHeaders))headers.set(name,value);
      const response=await (this.options.fetcher??fetch)(this.options.origin,{method:'POST',headers,
        body:fromBase64(frame.bodyBase64),redirect:'error',signal:operation.controller.signal});
      const body=await boundedBody(response,operation.controller.signal);
      if(!this.current(frame.id,operation))return;
      if(operation.controller.signal.aborted || Date.now()>=frame.deadlineAt){this.reply(frame,504,'origin_timeout');return;}
      const allowed=['content-type','mcp-protocol-version','retry-after'];
      const responseHeaders:[string,string][]=[];
      for(const name of allowed){const value=response.headers.get(name);if(value!==null)responseHeaders.push([name,value]);}
      const encoded=encodeRelayFrame({v:1,kind:'response',id:frame.id,generation:frame.generation,
        status:response.status,headers:responseHeaders,bodyBase64:toBase64(body)});
      if(encoded.ok)this.send(encoded.text);else this.reply(frame,502,'invalid_origin_response');
    } catch {
      if(this.current(frame.id,operation))this.reply(frame,operation.controller.signal.aborted?504:502,
        operation.controller.signal.aborted?'origin_timeout':'origin_unavailable');
    } finally {
      clearTimeout(operation.timer);
      if(this.operations.get(frame.id)===operation)this.operations.delete(frame.id);
    }
  }
  private current(id:string,operation:Operation):boolean {return !this.stopped && this.operations.get(id)===operation;}
  private reply(frame:RelayFrame,status:number,code:string):void {
    const encoded=encodeRelayFrame({v:1,kind:'response',id:frame.id,generation:frame.generation,status,
      headers:[['content-type','application/json']],bodyBase64:toBase64(new TextEncoder().encode(JSON.stringify({error:code})))});
    if(encoded.ok)this.send(encoded.text);else this.fail();
  }
  private send(text:string):void {try{this.options.send(text);}catch{this.fail();}}
  private fail():void {this.stop();this.options.close('connector_protocol_error');}
}
