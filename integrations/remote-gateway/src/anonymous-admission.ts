/** Coarse anonymous setup guard. Counters are per location, not a global quota. */
export interface SetupAdmissionEnv {ANON_SETUP?:RateLimit;ANON_GLOBAL?:RateLimit;}
export async function anonymousAdmission(request:Request,env:SetupAdmissionEnv):Promise<Response|null>{
 const path=new URL(request.url).pathname;
 if(!['/oauth/register','/bootstrap','/authorize','/owner-login','/consent','/select','/github/callback'].includes(path))return null;
 const unavailable=()=>Response.json({error:'setup_unavailable'},{status:503,headers:{'Cache-Control':'no-store','Retry-After':'60'}});
 if(!env.ANON_SETUP||!env.ANON_GLOBAL)return unavailable();
 try{
  const address=request.headers.get('CF-Connecting-IP')??'unknown';
  if(address.length>64||!/^[0-9a-fA-F:.]+$/.test(address)&&address!=='unknown')return unavailable();
  const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(address));
  const key=Array.from(new Uint8Array(digest),value=>value.toString(16).padStart(2,'0')).join('');
  if(!(await env.ANON_SETUP.limit({key})).success||!(await env.ANON_GLOBAL.limit({key:'setup'})).success)return Response.json({error:'setup_rate_limited'},{status:429,headers:{'Cache-Control':'no-store','Retry-After':'60'}});
  return null;
 }catch{return unavailable();}
}
