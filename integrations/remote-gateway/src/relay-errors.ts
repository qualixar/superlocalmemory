/** Fixed, non-secret relay failure codes a public MCP client may see. Each names
 * a distinct transport state, so a 503 can be diagnosed from the client side. */
const PUBLIC_RELAY_CODES=new Set([
 'connector_unavailable','connector_offline','connector_closed','connector_error','connector_replaced','stale_connector',
 'connection_revoked','upload_unsupported','relay_timeout','relay_busy','origin_timeout','request_cancelled','connector_asleep',
]);
/** Map a relay failure body to a public code; anything else is withheld. */
export function publicRelayCode(body:unknown):string {
 if(body&&typeof body==='object'&&'error' in body&&typeof body.error==='string'&&PUBLIC_RELAY_CODES.has(body.error))return body.error;
 return 'origin_unavailable';
}
