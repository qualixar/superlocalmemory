import {LocalRelaySession,connectorSocketOptions} from './local-connector.ts';
import type {RelayCodecOptions} from './relay-protocol.ts';

export interface EnrollmentCredential {
  endpoint:string; deviceToken:string; expiresAt:number;
  origin:string; originHeaders:Readonly<Record<string,string>>;
  codecOptions?:RelayCodecOptions;
}
export type ConnectorState='disabled'|'connecting'|'connected'|'reconnecting'|'stopped'|'authorization_required'|'configuration_error';
export interface ConnectorSocket {
  bufferedAmount:number;
  on(event:'message',callback:(data:{toString():string},binary:boolean)=>void):unknown;
  on(event:'close',callback:()=>void):unknown;
  on(event:'error',callback:(error:Error)=>void):unknown;
  on(event:'unexpected-response',callback:(request:unknown,response:{statusCode?:number;resume():void})=>void):unknown;
  send(text:string):void;
  terminate():void;
}
interface SupervisorOptions {
  enabled?:boolean;
  loadEnrollment:()=>Promise<EnrollmentCredential>;
  dial:(config:ReturnType<typeof connectorSocketOptions>)=>ConnectorSocket;
  onState:(state:ConnectorState)=>void;
  fetcher?:typeof fetch;
  retryMs?:number; readyTimeoutMs?:number; heartbeatMs?:number;
}

/** Background outbound-only transport. Enrollment/storage and public OAuth are
 * separate authorities. No replay of memory writes after a dropped connection.
 */
export class ConnectorSupervisor {
  private readonly options:SupervisorOptions;
  private running=false;
  private epoch=0;
  private failures=0;
  private cleanup:(()=>void)|null=null;
  private retry:ReturnType<typeof setTimeout>|null=null;

  constructor(options:SupervisorOptions){
    for(const value of [options.retryMs??1000,options.readyTimeoutMs??10000,options.heartbeatMs??20000])
      if(!Number.isSafeInteger(value)||value<=0||value>60000)throw new Error('invalid_supervisor_timing');
    this.options={...options};
  }
  private publish(state:ConnectorState):void{
    try{this.options.onState(state);}catch{console.error('connector_state_observer_failed');}
  }
  async start():Promise<void>{
    if(this.running)return;
    if(this.options.enabled!==true){this.publish('disabled');return;}
    this.running=true;const epoch=++this.epoch;await this.connect(epoch);
  }
  stop():void{
    this.running=false;this.epoch++;
    if(this.retry!==null){clearTimeout(this.retry);this.retry=null;}
    this.cleanup?.();this.cleanup=null;this.publish('stopped');
  }
  private current(epoch:number):boolean{return this.running&&this.epoch===epoch;}
  private halt(state:'authorization_required'|'configuration_error'):void{
    this.running=false;this.epoch++;
    if(this.retry!==null){clearTimeout(this.retry);this.retry=null;}
    this.cleanup?.();this.cleanup=null;this.publish(state);
  }
  private schedule(epoch:number):void{
    if(!this.current(epoch)||this.retry!==null)return;
    this.publish('reconnecting');
    const delay=Math.min(60000,(this.options.retryMs??1000)*2**Math.min(this.failures++,6));
    this.retry=setTimeout(()=>{this.retry=null;if(this.current(epoch))void this.connect(epoch).catch(()=>{
      if(this.current(epoch))this.halt('configuration_error');
    });},delay);
  }
  private async connect(epoch:number):Promise<void>{
    let credential:EnrollmentCredential, config:ReturnType<typeof connectorSocketOptions>;
    try {
      credential=await this.options.loadEnrollment();
      if(!this.current(epoch))return;
      if(!Number.isSafeInteger(credential.expiresAt)||credential.expiresAt<=Date.now()){this.halt('authorization_required');return;}
      config=connectorSocketOptions(credential.endpoint,credential.deviceToken);
    }catch{if(this.current(epoch))this.halt('configuration_error');return;}
    this.publish('connecting');
    let socket:ConnectorSocket;
    try{socket=this.options.dial(config);}catch{this.schedule(epoch);return;}
    try{this.attach(socket,credential,epoch);}catch{if(this.current(epoch))this.halt('configuration_error');}
  }
  private attach(socket:ConnectorSocket,credential:EnrollmentCredential,epoch:number):void{
    let closed=false, waitingPong=false;
    let session:LocalRelaySession;
    let heartbeat:ReturnType<typeof setInterval>|null=null;
    const timers:ReturnType<typeof setTimeout>[]=[];
    const clean=()=>{
      if(closed)return;closed=true;
      for(const timer of timers)clearTimeout(timer);
      if(heartbeat!==null)clearInterval(heartbeat);
      session?.stop();try{socket.terminate();}catch{console.error('connector_socket_shutdown_failed');}
    };
    const lost=()=>{if(closed)return;clean();this.schedule(epoch);};
    this.cleanup=clean;
    // ws can emit an asynchronous error when terminated during CONNECTING.
    // Register handlers before any validation/cleanup can terminate it.
    socket.on('close',lost);socket.on('error',lost);
    try{session=new LocalRelaySession({origin:credential.origin,originHeaders:credential.originHeaders,
      codecOptions:credential.codecOptions,fetcher:this.options.fetcher,
      send:text=>{if(socket.bufferedAmount>8*1024*1024)throw new Error('connector_backpressure');socket.send(text);},
      close:()=>lost()});}
    catch{this.halt('configuration_error');return;}
    const handshake=setTimeout(lost,this.options.readyTimeoutMs??10000);timers.push(handshake);
    // Timer chunks avoid overflow for long-lived credentials; wall clock is
    // checked again on every frame and every heartbeat, including after sleep.
    heartbeat=setInterval(()=>{
      if(credential.expiresAt<=Date.now()){this.halt('authorization_required');return;}
      if(!session.ready)return;
      if(waitingPong){lost();return;}waitingPong=true;
      try{socket.send('ping');}catch{lost();}
    },this.options.heartbeatMs??20000);
    socket.on('message',(data:{toString():string},binary:boolean)=>{
      if(!this.current(epoch)||closed)return;
      if(credential.expiresAt<=Date.now()){this.halt('authorization_required');return;}
      if(binary){lost();return;}
      const text=data.toString();
      if(text==='pong'&&waitingPong){waitingPong=false;return;}
      void session.receive(text).then(()=>{
        if(!this.current(epoch)||closed)return;
        if(session.ready){clearTimeout(handshake);this.failures=0;this.publish('connected');}
      }).catch(()=>lost());
    });
    socket.on('unexpected-response',(_request:unknown,response:{statusCode?:number;resume():void})=>{
      response.resume();
      if(!this.current(epoch)||closed)return;
      if(response.statusCode===401||response.statusCode===403)this.halt('authorization_required');else lost();
    });
  }
}
