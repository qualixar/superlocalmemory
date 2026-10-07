import {visit} from 'jsonc-parser';
import type {HeaderPair} from './relay-protocol.ts';
import {MAX_REQUEST_BYTES,MAX_RESPONSE_BYTES} from './relay-protocol.ts';
import type {RequestEnvelope} from './contracts.ts';

export const SUPPORTED_VERSIONS=['2026-07-28','2025-11-25','2025-06-18','2025-03-26','2024-11-05'] as const;
export class GatewayInputError extends Error {
  readonly code:string;readonly status:number;readonly rpcCode:number;readonly rpcId:string|number|null;readonly data?:unknown;
  constructor(code:string,status=400,rpcCode=-32600,rpcId:string|number|null=null,data?:unknown){super(code);this.code=code;this.status=status;this.rpcCode=rpcCode;this.rpcId=rpcId;this.data=data;}
}
export interface ParameterHeader {tool:string;header:string;path:readonly string[];type?:'string'|'integer'|'boolean';}
interface ParseOptions {parameterHeaders?:readonly ParameterHeader[];}
export interface ParsedMcpRequest {envelope:RequestEnvelope;headers:HeaderPair[];parameterHeaderNames:string[];}
const record=(v:unknown):v is Record<string,unknown>=>v!==null&&typeof v==='object'&&!Array.isArray(v);

function strictJson(text:string):unknown {
  const stack:(Set<string>|null)[]=[];let invalid=false;
  try {
    visit(text,{
      onObjectBegin(){if(stack.length>=64)throw new Error('depth');stack.push(new Set());},
      onObjectProperty(name){const seen=stack.at(-1);if(!seen||seen.has(name)){invalid=true;return;}seen.add(name);},
      onObjectEnd(){stack.pop();},onArrayBegin(){if(stack.length>=64)throw new Error('depth');stack.push(null);},onArrayEnd(){stack.pop();},
      onError(){invalid=true;},
    },{disallowComments:true,allowTrailingComma:false,allowEmptyContent:false});
    if(invalid)throw new Error('invalid_json');
    return JSON.parse(text);
  }catch{throw new GatewayInputError('INVALID_JSON',400,-32700);}
}

async function boundedBody(request:Request):Promise<Uint8Array> {
  if(!request.body)return new Uint8Array();
  const reader=request.body.getReader();const chunks:Uint8Array[]=[];let total=0,timedOut=false;
  const cancel=()=>{void reader.cancel().catch(()=>{});};
  const timer=setTimeout(()=>{timedOut=true;cancel();},5000);
  request.signal.addEventListener('abort',cancel,{once:true});
  try {
    for(;;){const part=await reader.read();if(timedOut||request.signal.aborted)throw new GatewayInputError('REQUEST_TIMEOUT',408);if(part.done)break;
      total+=part.value.length;if(total>MAX_REQUEST_BYTES){cancel();throw new GatewayInputError('BODY_TOO_LARGE',413);}chunks.push(part.value);}
  }finally{clearTimeout(timer);request.signal.removeEventListener('abort',cancel);reader.releaseLock();}
  const result=new Uint8Array(total);let offset=0;for(const chunk of chunks){result.set(chunk,offset);offset+=chunk.length;}return result;
}
function headerValue(value:string):string {
  if(value.length>8192||/[^\x20-\x7e]/.test(value))throw new GatewayInputError('HEADER_MISMATCH',400,-32020);
  if(value.startsWith('=?base64?')&&value.endsWith('?=')){
    try{const encoded=value.slice(9,-2);const binary=atob(encoded);if(btoa(binary)!==encoded)throw new Error('base64');
      return new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(Uint8Array.from(binary,c=>c.charCodeAt(0)));}
    catch{throw new GatewayInputError('HEADER_MISMATCH',400,-32020);}
  }
  if(value.trim()!==value)throw new GatewayInputError('HEADER_MISMATCH',400,-32020);
  return value;
}
function parameters(options:ParseOptions):readonly ParameterHeader[] {
  const list=options.parameterHeaders??[];const seen=new Set<string>();
  if(!Array.isArray(list)||list.length>32)throw new GatewayInputError('INVALID_SCHEMA_CONFIGURATION',503);
  for(const item of list){if(!item||typeof item.tool!=='string'||!/^mcp-param-[a-z0-9_.-]{1,64}$/i.test(item.header)||!Array.isArray(item.path)||!item.path.length||item.path.length>16||item.path.some((p:unknown)=>typeof p!=='string'||!p||['__proto__','prototype','constructor'].includes(p))||seen.has(item.header.toLowerCase())||(item.type!==undefined&&!['string','integer','boolean'].includes(item.type)))throw new GatewayInputError('INVALID_SCHEMA_CONFIGURATION',503);seen.add(item.header.toLowerCase());}
  return list;
}
function metadata(request:Request,message:Record<string,unknown>,params:Record<string,unknown>,tool:string|undefined,annotations:readonly ParameterHeader[]):{modern:boolean;names:string[]} {
  const header=request.headers.get('mcp-protocol-version');const meta=record(params._meta)?params._meta:{};
  const body=meta['io.modelcontextprotocol/protocolVersion'];const modern=header==='2026-07-28'||body==='2026-07-28';
  if(header&&!SUPPORTED_VERSIONS.includes(header as typeof SUPPORTED_VERSIONS[number]))throw new GatewayInputError('UNSUPPORTED_PROTOCOL_VERSION',400,-32022,null,{requested:/^\d{4}-\d{2}-\d{2}$/.test(header)?header:'invalid',supported:SUPPORTED_VERSIONS});
  if(modern&&(header!=='2026-07-28'||body!==header||request.headers.get('mcp-method')!==message.method))throw new GatewayInputError('HEADER_MISMATCH',400,-32020);
  const method=request.headers.get('mcp-method');if(method!==null&&method!==message.method)throw new GatewayInputError('HEADER_MISMATCH',400,-32020);
  const name=request.headers.get('mcp-name');if((modern&&tool&&name===null)||(name!==null&&headerValue(name)!==tool))throw new GatewayInputError('HEADER_MISMATCH',400,-32020);
  const args=record(params.arguments)?params.arguments:{};const names:string[]=[];
  for(const annotation of annotations.filter(a=>a.tool===tool)){
    let value:unknown=args;for(const path of annotation.path){value=record(value)&&Object.hasOwn(value,path)?value[path]:undefined;}
    const actual=request.headers.get(annotation.header);
    if(value===undefined||value===null){if(actual!==null)throw new GatewayInputError('HEADER_MISMATCH',400,-32020);continue;}
    if(!['string','number','boolean'].includes(typeof value)||(typeof value==='number'&&!Number.isSafeInteger(value))||(annotation.type!==undefined&&(annotation.type==='integer'?typeof value!=='number':typeof value!==annotation.type)))throw new GatewayInputError('HEADER_MISMATCH',400,-32020);
    if((modern&&actual===null)||(actual!==null&&headerValue(actual)!==String(value)))throw new GatewayInputError('HEADER_MISMATCH',400,-32020);
    if(actual!==null)names.push(annotation.header.toLowerCase());
  }
  // This resource serves a curated tool schema. Unknown annotations require a
  // fresh tools/list instead of selecting routing authority from caller headers.
  for(const [key] of request.headers)if(key.startsWith('mcp-param-')&&!names.includes(key))throw new GatewayInputError('HEADER_MISMATCH',400,-32020);
  return {modern,names};
}
export async function parseMcpRequest(request:Request,options:ParseOptions={}):Promise<ParsedMcpRequest> {
  if(request.method!=='POST')throw new GatewayInputError('METHOD_NOT_ALLOWED',405);
  if(request.headers.get('content-type')?.split(';')[0]?.trim().toLowerCase()!=='application/json')throw new GatewayInputError('JSON_REQUIRED',415);
  const bytes=await boundedBody(request);let text:string;try{text=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(bytes);}catch{throw new GatewayInputError('INVALID_UTF8');}
  const message=strictJson(text);
  if(!record(message)||message.jsonrpc!=='2.0'||typeof message.method!=='string'||message.method.length>128||Object.hasOwn(message,'result')||Object.hasOwn(message,'error'))throw new GatewayInputError('INVALID_JSONRPC');
  const rpcId=message.id;if(rpcId!==undefined&&rpcId!==null&&!(typeof rpcId==='string'&&rpcId.length<=128)&&!(typeof rpcId==='number'&&Number.isSafeInteger(rpcId)))throw new GatewayInputError('INVALID_JSONRPC');
  if(message.params!==undefined&&!record(message.params))throw new GatewayInputError('INVALID_PARAMS');
  const params=record(message.params)?message.params:{};
  if(params.arguments!==undefined&&!record(params.arguments))throw new GatewayInputError('INVALID_ARGUMENTS');
  const tool=message.method==='tools/call'?params.name:undefined;
  if(message.method==='tools/call'&&(typeof tool!=='string'||!tool||tool.length>128))throw new GatewayInputError('INVALID_TOOL_NAME');
  const annotation=parameters(options);const {modern,names}=metadata(request,message,params,tool as string|undefined,annotation);
  const permitted=new Set(['content-type','accept','mcp-protocol-version','mcp-method','mcp-name',...names]);
  const headers:HeaderPair[]=[...request.headers].filter(([name])=>permitted.has(name));
  return {envelope:{era:modern?'modern-2026-07-28':'legacy',rpcMethod:message.method,rpcId:rpcId as string|number|null|undefined,toolName:tool as string|undefined,arguments:params.arguments as Record<string,unknown>|undefined,originalBody:bytes},headers,parameterHeaderNames:names};
}
export function filterMcpResponse(bytes:Uint8Array,request:Pick<RequestEnvelope,'rpcMethod'|'rpcId'>,allowedTools:readonly string[]):string {
  if(bytes.byteLength>MAX_RESPONSE_BYTES)throw new GatewayInputError('ORIGIN_RESPONSE_TOO_LARGE',502);
  let text:string;try{text=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(bytes);}catch{throw new GatewayInputError('INVALID_ORIGIN_RESPONSE',502);}
  const response=strictJson(text);
  if(!record(response)||response.jsonrpc!=='2.0'||(response.id!==request.rpcId&&!(response.id===null&&Object.hasOwn(response,'error'))))throw new GatewayInputError('INVALID_ORIGIN_RESPONSE',502);
  if(Object.hasOwn(response,'error'))return JSON.stringify({jsonrpc:'2.0',id:response.id,error:{code:record(response.error)&&Number.isSafeInteger(response.error.code)?response.error.code:-32603,message:'The memory request could not be completed.'}});
  if(!record(response.result))throw new GatewayInputError('INVALID_ORIGIN_RESPONSE',502);
  if(request.rpcMethod==='tools/list'){
    if(!Array.isArray(response.result.tools))throw new GatewayInputError('INVALID_ORIGIN_RESPONSE',502);
    const tools=response.result.tools.filter(t=>record(t)&&typeof t.name==='string'&&allowedTools.includes(t.name));
    return JSON.stringify({...response,result:{...response.result,tools,cacheScope:'private',ttlMs:0}});
  }
  if(request.rpcMethod==='server/discover'){
    if(!Array.isArray(response.result.supportedVersions))throw new GatewayInputError('INVALID_ORIGIN_RESPONSE',502);
    return JSON.stringify({...response,result:{resultType:'complete',supportedVersions:response.result.supportedVersions,capabilities:{tools:{}},instructions:'Scoped SuperLocalMemory access. Use tools/list for available tools.',cacheScope:'private',ttlMs:0}});
  }
  if(request.rpcMethod==='initialize')return JSON.stringify({...response,result:{...response.result,capabilities:{tools:{}},instructions:'Scoped SuperLocalMemory access. Use tools/list for available tools.'}});
  return text;
}
