/**
 * Account-owner cleanup initiated by explicit consent on the GitHub-authenticated
 * OAuth selection page. This helper is never exposed as an unauthenticated API.
 */
import type {OwnedConnection,OwnerIndexDO} from './owner-index-do.ts';

interface OwnerCleanupEnv {
 REGISTRIES: {getByName(id:string):{revokeConnection(ownerId:string,generation:number):Promise<number|void>}};
 DEVICES: {getByName(id:string):{revoke(ownerId:string):Promise<void>}};
 RELAYS: {getByName(id:string):{revoke():Promise<void>}};
}
type OwnerOps=Pick<OwnerIndexDO,'list'|'getConnection'|'cancelConnection'|'revoke'|'markCleanup'>;

/**
 * Preserve selected current connection and any other host/profile pair. The
 * caller is responsible for verifying OAuth owner identity, CSRF and explicit
 * opt-in before entering this function.
 */
export async function pruneOtherOwnerConnections({owner,ownerId,selectedId,env}:{
 owner:OwnerOps;ownerId:string;selectedId:string;env:OwnerCleanupEnv;
}):Promise<string[]>{
 if(!/^[a-f0-9]{32}$/.test(selectedId))throw new Error('invalid_selected_connection');
 const selected=await owner.getConnection(ownerId,selectedId);
 if(!selected||selected.revokedAt!==null)throw new Error('selected_connection_unavailable');
 const all=await owner.list(ownerId);
 const obsolete=all.filter(row=>row.connectionId!==selectedId&&row.revokedAt===null&&
   row.profileId===selected.profileId&&row.host===selected.host);
 const revoked:string[]=[];
 for(const row of obsolete){
   // The owner Directory is authoritative: changes between listing and
   // revocation are fenced via its current generation.
   const live=await owner.getConnection(ownerId,row.connectionId);
   if(!live||live.revokedAt!==null||live.profileId!==selected.profileId||
      live.host!==selected.host)continue;
   await owner.cancelConnection(ownerId,live.connectionId);
   const generation=await owner.revoke(ownerId,live.connectionId,live.generation);
   await env.REGISTRIES.getByName(live.connectionId).revokeConnection(ownerId,1);
   // A device the kept connection still uses is never revoked with the old one.
   if(live.deviceDigest!==selected.deviceDigest)await env.DEVICES.getByName(live.deviceDigest).revoke(ownerId);
   await env.RELAYS.getByName(live.connectionId).revoke();
   await owner.markCleanup(ownerId,live.connectionId,generation);
   revoked.push(live.connectionId);
 }
 return revoked;
}
