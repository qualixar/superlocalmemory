/** Byte-bounded, time-bounded UTF8 input for public authorization endpoints. */
export async function readAuthorizationBody(request:Request,options:{limit?:number;timeoutMs?:number}={}):Promise<string>{
 const limit=options.limit??16384;const timeoutMs=options.timeoutMs??5000;
 if(!request.body)return '';
 const reader=request.body.getReader();const chunks:Uint8Array[]=[];let length=0;let expired=false;
 let timer:ReturnType<typeof setTimeout>;
 const deadline=new Promise<never>((_,reject)=>{timer=setTimeout(()=>{expired=true;void reader.cancel().catch(()=>{});reject(new Error('body_timeout'));},timeoutMs);});
 try{
  for(;;){const part=await Promise.race([reader.read(),deadline]);if(expired)throw new Error('body_timeout');if(part.done)break;length+=part.value.length;if(length>limit)throw new Error('body_too_large');chunks.push(part.value);}
  const bytes=new Uint8Array(length);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
  try{return new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(bytes);}catch{throw new Error('invalid_body');}
 }finally{clearTimeout(timer!);void reader.cancel().catch(()=>{});reader.releaseLock();}
}
