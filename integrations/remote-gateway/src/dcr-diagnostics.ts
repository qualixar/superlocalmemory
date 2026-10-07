/** Opt-in, short-lived protocol diagnostics. Never retain credential values. */
const fields=['client_name','redirect_uris','client_uri','logo_uri','policy_uri','tos_uri','jwks_uri','contacts','grant_types','response_types','token_endpoint_auth_method','token_endpoint_auth_methods_supported'];
const errors=new Set(['invalid_request','invalid_client_metadata','server_error','temporarily_unavailable']);
const methods=new Set(['none','client_secret_basic','client_secret_post']);
const hosts=new Set(['backend.composio.dev','dashboard.composio.dev','127.0.0.1','localhost']);
export function registrationDiagnostic(input:unknown,status:number,outcome:unknown){
 const data=input&&typeof input==='object'&&!Array.isArray(input)?input as Record<string,unknown>:{};
 const result=outcome&&typeof outcome==='object'?outcome as Record<string,unknown>:{};
 const description=typeof result.error_description==='string'?result.error_description:'';
 const invalid=/^Invalid ([a-z_]+):/.exec(description)?.[1];
 return {status,error:typeof result.error==='string'&&errors.has(result.error)?result.error:'unclassified',invalidField:invalid&&fields.includes(invalid)?invalid:null,
  method:data.token_endpoint_auth_method===undefined?'default':typeof data.token_endpoint_auth_method==='string'&&methods.has(data.token_endpoint_auth_method)?data.token_endpoint_auth_method:'unsupported',
  fields:Object.fromEntries(fields.map(field=>[field,data[field]===undefined?'missing':data[field]===null?'null':Array.isArray(data[field])?'array':typeof data[field]==='string'?(data[field]===''?'empty-string':'string'):typeof data[field]])),
  redirects:Array.isArray(data.redirect_uris)?data.redirect_uris.slice(0,16).map(value=>{try{const uri=new URL(String(value));return {scheme:['https:','http:'].includes(uri.protocol)?uri.protocol:'other',host:hosts.has(uri.hostname)?uri.hostname:'other',userinfo:!!(uri.username||uri.password),fragment:!!uri.hash};}catch{return {invalid:true};}}):[],
  grantTypes:Array.isArray(data.grant_types)?data.grant_types.slice(0,16).map(value=>typeof value==='string'&&['authorization_code','refresh_token'].includes(value)?value:'unsupported'):[]};
}
