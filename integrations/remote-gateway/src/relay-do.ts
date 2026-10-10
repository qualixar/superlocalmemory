import { DurableObject } from "cloudflare:workers";
import { GRANT_HEADER, importGrantKey, signGrant, toBase64Url, unwrapGrantKey, wrapGrantKey, type GrantInput, type StoredGrantKey } from "./grant.ts";
import { decodeRelayFrame, encodeRelayFrame, RELAY_DEADLINE_MS, type RelayFrame, type RelayCodecOptions } from "./relay-protocol.ts";
export interface RelayBinding {
  ownerId: string; connectionId: string; installationId: string; profileId: string;
  deviceDigest: string; deviceExpiresAt: number;
}
/** The laptop sends a heartbeat after 20 s without traffic. Silence longer than two
 * missed heartbeats means it is asleep or its network dropped without closing. */
export const CONNECTOR_SILENCE_MS = 45000;
interface StoredState { binding: RelayBinding | null; generation: number; revoked: boolean; }
interface Attachment { generation: number; connectionId: string; connectedAt?: number; }
/** What the resource Worker passes with a forwarded call: who is calling (signed for the laptop) and whether it parks on the laptop. */
export interface ForwardContext { grant?: GrantInput; wait?: boolean; }
/** A laptop holds one slot per long wait; more than this many at once would starve ordinary calls. */
const MAX_CONCURRENT_WAITS = 2;
interface Pending { callerId: string; wait: boolean; socket: WebSocket; generation: number; settle: (response: Response) => void; timer: ReturnType<typeof setTimeout>; }
function failure(status: number, code: string): Response {
  return Response.json({ error: code }, { status, headers: { "Cache-Control": "no-store" } });
}
function validBinding(value: RelayBinding): boolean {
  if (!value || typeof value !== "object" || Object.keys(value).length !== 6) return false;
  return [value.ownerId,value.connectionId,value.installationId,value.profileId].every(id=>typeof id==="string" && id.length>0 && id.length<=256 && id.trim()===id) &&
    typeof value.deviceDigest==="string" && /^[a-f0-9]{64}$/.test(value.deviceDigest) && Number.isSafeInteger(value.deviceExpiresAt) && value.deviceExpiresAt>=0;
}
function decodeBody(text: string): Uint8Array<ArrayBuffer> {
  const binary=atob(text);const bytes=new Uint8Array(binary.length);
  for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
  return bytes;
}
/** Private namespace component. configureBinding/revoke/forward require a trusted
 * authenticated control/resource Worker. No public production routing supplied.
 * Public OAuth admission/profile policy remains a separate composed-path gate.
 */
export class RelayDO extends DurableObject {
  private state: StoredState={binding:null,generation:0,revoked:false};
  private pending=new Map<string,Pending>();
  private heard=new Map<WebSocket,number>();
  /** Sockets accepted before connection times were recorded count from here. */
  private readonly startedAt=Date.now();
  private signing:{version:number;key:CryptoKey}|null=null;
  /** True once storage was read and held no key, so older laptops cost no read per call. */
  private unkeyed=false;
  constructor(ctx: DurableObjectState,env: Cloudflare.Env) {
    super(ctx,env);
    this.ctx.blockConcurrencyWhile(async()=>{
      const saved=await this.ctx.storage.get<StoredState>("relay-state");
      if(saved){if((saved.binding!==null&&!validBinding(saved.binding))||!Number.isSafeInteger(saved.generation)||saved.generation<0||typeof saved.revoked!=="boolean")throw new Error("invalid_relay_state");this.state=saved;}
    });
    this.ctx.setWebSocketAutoResponse(new WebSocketRequestResponsePair("ping","pong"));
  }
  async configureBinding(binding: RelayBinding): Promise<void> {
    if(!validBinding(binding))throw new Error("invalid_binding");
    const error=await this.ctx.blockConcurrencyWhile(async():Promise<string|null>=>{
      const current=this.state.binding;
      if(this.state.revoked)return "connection_revoked";
      if(current && ["ownerId","connectionId","installationId","profileId"].some(key=>current[key as keyof RelayBinding]!==binding[key as keyof RelayBinding]))return "binding_conflict";
      const rotating=current!==null && current.deviceDigest!==binding.deviceDigest;
      if(rotating && this.state.generation>=Number.MAX_SAFE_INTEGER)return "generation_exhausted";
      const next={...this.state,binding:{...binding},generation:this.state.generation+(rotating?1:0)};
      // Durable commit precedes in-memory publication or any success response.
      await this.ctx.storage.put("relay-state",next);this.state=next;
      if(rotating)for(const socket of this.ctx.getWebSockets("connector"))this.closeSocket(socket,503,"device_rotated");
      return null;
    });
    // Expected business errors outside the input gate must not reset the DO.
    if(error)throw new Error(error);
  }
  async revoke(): Promise<void> {
    await this.ctx.blockConcurrencyWhile(async()=>{
      if(this.state.generation>=Number.MAX_SAFE_INTEGER)throw new Error("generation_exhausted");
      const next={...this.state,revoked:true,generation:this.state.generation+1};
      await this.ctx.storage.put("relay-state",next);this.state=next;
      await this.ctx.storage.delete("grant-key");this.signing=null;this.unkeyed=true;
      for(const socket of this.ctx.getWebSockets("connector"))this.closeSocket(socket,403,"connection_revoked");
    });
  }
  private wrapSecret(): string|null {
    const secret=(this.env as {GRANT_WRAP_KEY?:unknown}).GRANT_WRAP_KEY;
    return typeof secret==="string"&&/^[a-fA-F0-9]{64}$/.test(secret)?secret:null;
  }
  /** Mints the connection's grant key and returns it in clear exactly once, to the owner's own authenticated channel. Every call replaces the key. */
  async rotateGrantKey(ownerId: string): Promise<{version:number;key:string}> {
    const result=await this.ctx.blockConcurrencyWhile(async():Promise<{value:{version:number;key:string}}|{error:string}>=>{
      if(this.state.revoked)return {error:"connection_revoked"};
      const binding=this.state.binding;
      if(!binding)return {error:"connection_unconfigured"};
      if(binding.ownerId!==ownerId)return {error:"owner_mismatch"};
      const secret=this.wrapSecret();
      if(!secret)return {error:"grant_unavailable"};
      const previous=await this.ctx.storage.get<StoredGrantKey>("grant-key");
      const version=(Number.isSafeInteger(previous?.version)?previous!.version:0)+1;
      const raw=crypto.getRandomValues(new Uint8Array(32));
      await this.ctx.storage.put("grant-key",await wrapGrantKey(raw,secret,version));
      this.signing={version,key:await importGrantKey(raw)};this.unkeyed=false;
      return {value:{version,key:toBase64Url(raw)}};
    });
    if("error" in result)throw new Error(result.error);
    return result.value;
  }
  /** The key in force, or null when none was ever minted or it cannot be read: callers then send no grant at all. */
  private async signingKey(): Promise<{version:number;key:CryptoKey}|null> {
    if(this.signing)return this.signing;
    if(this.unkeyed)return null;
    const secret=this.wrapSecret();const stored=await this.ctx.storage.get<StoredGrantKey>("grant-key");
    if(!stored){this.unkeyed=true;return null;}
    if(!secret)return null;
    try{this.signing={version:stored.version,key:await importGrantKey(await unwrapGrantKey(stored,secret))};}catch{return null;}
    return this.signing;
  }
  async forwardCurrent(request: Omit<Extract<RelayFrame,{kind:'request'}>,'generation'>, options: RelayCodecOptions = {}, context: ForwardContext = {}): Promise<Response> {
    // Only the authenticated resource Worker calls this. Client metadata cannot
    // choose a socket generation; forward() still checks current attachment.
    return this.forward({...request,generation:Math.max(1,this.state.generation)},options,context);
  }
  async cancelCaller(identifier:string):Promise<void> {
    for(const [wireId,pending] of this.pending){
      if(pending.callerId!==identifier)continue;
      const encoded=encodeRelayFrame({v:1,kind:'cancel',id:wireId,generation:pending.generation});
      if(encoded.ok){try{pending.socket.send(encoded.text);}catch{}}
      this.finish(wireId,failure(499,'request_cancelled'));
    }
  }
  async fetch(request: Request): Promise<Response> {
    if(new URL(request.url).pathname!=="/connector")return failure(404,"not_found");
    if(request.method!=="GET")return failure(405,"method_not_allowed");
    if(request.headers.get("Upgrade")?.toLowerCase()!=="websocket")return failure(426,"upgrade_required");
    return this.ctx.blockConcurrencyWhile(async()=>{
      const binding=this.state.binding;
      if(this.state.revoked)return failure(403,"connection_revoked");
      if(!binding)return failure(503,"connection_unconfigured");
      if(binding.deviceExpiresAt<=Date.now())return failure(401,"device_expired");
      const match=/^Bearer ([A-Za-z0-9_-]{32,256})$/.exec(request.headers.get("Authorization")??"");
      if(!match)return failure(401,"device_unauthorized");
      const digest=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(match[1]));
      const expected=new Uint8Array(binding.deviceDigest.match(/../g)!.map(byte=>parseInt(byte,16)));
      if(!crypto.subtle.timingSafeEqual(digest,expected))return failure(401,"device_unauthorized");
      if(this.state.generation>=Number.MAX_SAFE_INTEGER)return failure(503,"generation_exhausted");
      const next={...this.state,generation:this.state.generation+1};
      await this.ctx.storage.put("relay-state",next);this.state=next;
      for(const old of this.ctx.getWebSockets("connector"))this.closeSocket(old,503,"connector_replaced");
      const pair=new WebSocketPair();const [client,server]=Object.values(pair);
      this.ctx.acceptWebSocket(server,["connector"]);
      server.serializeAttachment({generation:next.generation,connectionId:binding.connectionId,connectedAt:Date.now()} satisfies Attachment);
      server.send(JSON.stringify({v:1,kind:"ready",generation:next.generation}));
      return new Response(null,{status:101,webSocket:client});
    });
  }
  async forward(callerFrame: RelayFrame,options: RelayCodecOptions={},context: ForwardContext={}): Promise<Response> {
    // The grant header is the relay's alone: whatever a caller put there is discarded.
    const frame=callerFrame.kind==="request"?{...callerFrame,headers:callerFrame.headers.filter(pair=>pair[0].toLowerCase()!==GRANT_HEADER)}:callerFrame;
    if(this.state.revoked)return failure(403,"connection_revoked");
    if(!this.state.binding||this.state.binding.deviceExpiresAt<=Date.now())return failure(503,"connector_unavailable");
    const encoded=encodeRelayFrame(frame,options);
    if(!encoded.ok||frame.kind!=="request")return failure(400,"invalid_relay_request");

    const remaining=frame.deadlineAt-Date.now();
    if(remaining<=0)return failure(504,"relay_timeout");
    if(remaining>RELAY_DEADLINE_MS)return failure(400,"invalid_deadline");
    const socket=this.currentSocket();
    if(!socket)return failure(503,"connector_offline");
    if(frame.generation!==this.state.generation)return failure(409,"stale_generation");
    if(Date.now()-this.lastHeard(socket)>CONNECTOR_SILENCE_MS)return failure(503,"connector_asleep");
    // Fresh wire nonce even if a caller reuses its ID after timeout. A late reply
    // can never complete a later operation that happens to reuse that caller ID.
    const wireId=crypto.randomUUID();
    // Signing may wait on storage, so it happens before the capacity checks below, which must run without an await in between.
    const binding=this.state.binding;const key=context.grant?await this.signingKey():null;
    const headers=key&&context.grant?[...frame.headers,[GRANT_HEADER,await signGrant(key.key,key.version,context.grant,{cid:binding.connectionId,fid:wireId,gen:frame.generation,dl:frame.deadlineAt})] as const]:frame.headers;
    const outbound=encodeRelayFrame({...frame,id:wireId,headers},options);
    if(!outbound.ok)return failure(400,"invalid_relay_request");
    if(this.state.revoked||!this.state.binding||this.currentSocket()!==socket)return failure(503,"connector_offline");
    const wait=context.wait===true;
    if(this.pending.size>=8||(wait&&[...this.pending.values()].filter(p=>p.wait).length>=MAX_CONCURRENT_WAITS))return failure(429,"relay_busy");
    if([...this.pending.values()].some(p=>p.callerId===frame.id))return failure(409,"duplicate_request");
    return new Promise<Response>(resolve=>{
      const timer=setTimeout(()=>{
        const entry=this.pending.get(wireId);if(!entry)return;
        this.pending.delete(wireId);
        const cancel=encodeRelayFrame({v:1,kind:"cancel",id:wireId,generation:entry.generation});
        try{if(cancel.ok)socket.send(cancel.text);}catch{/* transport already gone; do not claim cancellation reached the origin */}
        resolve(failure(504,"relay_timeout"));
      },remaining);
      this.pending.set(wireId,{callerId:frame.id,wait,socket,generation:frame.generation,settle:resolve,timer});
      try{socket.send(outbound.text);}catch{this.finish(wireId,failure(503,"connector_offline"));}
    });
  }
  webSocketMessage(socket: WebSocket,message: string|ArrayBuffer): void {
    if(this.state.revoked||!this.state.binding||this.state.binding.deviceExpiresAt<=Date.now()){this.closeSocket(socket,403,"connection_unavailable");return;}
    const attachment=socket.deserializeAttachment() as Attachment|null;
    if(!attachment||attachment.generation!==this.state.generation||attachment.connectionId!==this.state.binding.connectionId){this.closeSocket(socket,503,"stale_connector");return;}
    this.heard.set(socket,Date.now());
    const decoded=decodeRelayFrame(typeof message==="string"?message:new Uint8Array(message));
    if(!decoded.ok||decoded.frame.kind!=="response"){this.closeSocket(socket,400,"invalid_connector_frame");return;}
    const frame=decoded.frame;const entry=this.pending.get(frame.id);
    if(!entry||entry.socket!==socket||entry.generation!==frame.generation)return;
    const bodyless=frame.status===204||frame.status===205||frame.status===304;
    if(bodyless && frame.bodyBase64!==""){this.closeSocket(socket,400,"invalid_response_body");return;}
    this.finish(frame.id,new Response(bodyless?null:decodeBody(frame.bodyBase64),{status:frame.status,headers:new Headers(frame.headers.map(pair=>[pair[0],pair[1]]))}));
  }
  webSocketClose(socket: WebSocket): void { this.closeSocket(socket,503,"connector_closed"); }
  webSocketError(socket: WebSocket): void { this.closeSocket(socket,503,"connector_error"); }
  /** Latest sign of life: an automatic heartbeat reply (kept across hibernation),
   * a message from the laptop, or the moment the socket connected. */
  private lastHeard(socket: WebSocket): number {
    const attachment=socket.deserializeAttachment() as Attachment|null;
    const heartbeat=this.ctx.getWebSocketAutoResponseTimestamp(socket)?.getTime()??0;
    return Math.max(heartbeat,this.heard.get(socket)??0,attachment?.connectedAt??this.startedAt);
  }
  private currentSocket(): WebSocket|null {
    return this.ctx.getWebSockets("connector").find(socket=>{const a=socket.deserializeAttachment() as Attachment|null;return a?.generation===this.state.generation&&a.connectionId===this.state.binding?.connectionId;})??null;
  }
  private finish(id: string,response: Response): void {
    const entry=this.pending.get(id);if(!entry)return;
    this.pending.delete(id);clearTimeout(entry.timer);entry.settle(response);
  }
  private closeSocket(socket: WebSocket,status: number,code: string): void {
    for(const [id,entry] of this.pending)if(entry.socket===socket)this.finish(id,failure(status,code));
    this.heard.delete(socket);
    if(socket.readyState===WebSocket.OPEN)socket.close(1000,"connector_closed");
  }
}
