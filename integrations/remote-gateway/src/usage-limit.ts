/** Whole seconds until the daily tool-call quota resets (next 00:00 UTC), at least 1. */
export function secondsUntilUtcMidnight(nowMs:number):number {
 const now=new Date(nowMs);
 const next=Date.UTC(now.getUTCFullYear(),now.getUTCMonth(),now.getUTCDate()+1);
 return Math.max(1,Math.ceil((next-nowMs)/1000));
}
