/** GitHub-account-scoped SLM connection picker, with explicit owner cleanup. */
type Connection = {
 connectionId:string; profileId:string; host:string; deviceExpiresAtMs:number;
};
const htmlEscape=(value:string):string=>value.replaceAll('&','&amp;').replaceAll('<','&lt;')
 .replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');

export function renderConnectionSelector(
 connections:readonly Connection[],handle:string,scopes:readonly string[]
):string {
 const ordered=[...connections].sort((a,b)=>b.deviceExpiresAtMs-a.deviceExpiresAtMs);
 const options=ordered.map(c=>{
   const expiry=Number.isFinite(c.deviceExpiresAtMs)?
     new Date(c.deviceExpiresAtMs).toISOString().slice(0,16).replace('T',' '):'unknown';
   const label=c.profileId+' ('+c.host+') — ID '+c.connectionId.slice(0,8)+' — expires '+expiry+' UTC';
   return '<option value="'+htmlEscape(c.connectionId)+'">'+htmlEscape(label)+'</option>';
 }).join('');
 const cleanup=ordered.length>1?
   '<label class="note"><input type="checkbox" name="remove_duplicates" value="yes"> Remove other connections with the same local profile and app from this GitHub account. This revokes access for older connected installations. Leave unchecked to keep all connections.</label>':'';
 return '<h1>Choose your SLM connection</h1><p>GitHub sign-in is complete. Select the local profile this application may use. The ID distinguishes otherwise identical connections; verify the current one in your local SLM dashboard.</p>'+
 '<form method="post" action="/select"><input type="hidden" name="handle" value="'+htmlEscape(handle)+'"><label>Local profile <select name="connection_id">'+options+'</select></label>'+
 cleanup+'<div class="actions"><button name="decision" value="allow">Connect</button><button name="decision" value="deny">Cancel</button></div></form>'+
 '<details><summary>Approved permissions</summary><p>'+scopes.map(htmlEscape).join(', ')+'</p></details>';
}
