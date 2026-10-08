import {expect,test} from 'vitest';
import {exchangeGithubCode} from '../../src/auth-flow.ts';
import {verifyGithubIdentity} from '../../src/authorization-policy.ts';
test('GitHub credential exchange constructs a valid actual workerd request without following redirects',async()=>{
 const value=await exchangeGithubCode('client','synthetic','synthetic-code','a'.repeat(43),async(input,options)=>{const request=new Request(input,options);expect(request.redirect).toBe('manual');return Response.json({access_token:'synthetic-token'});});expect(value).toBe('synthetic-token');
});
test('GitHub identity lookup constructs a valid actual workerd request and denies redirects',async()=>{
 expect(await verifyGithubIdentity('synthetic-token',async(input,options)=>{const request=new Request(input,options);expect(request.redirect).toBe('manual');return Response.json({id:16027584});})).toBe('16027584');
 await expect(verifyGithubIdentity('synthetic-token',async()=>new Response(JSON.stringify({id:16027584}),{status:302,headers:{Location:'https://evil.example/'}}))).rejects.toThrow();
});
