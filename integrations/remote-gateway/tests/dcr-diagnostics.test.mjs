import test from 'node:test';import assert from 'node:assert/strict';
import {registrationDiagnostic} from '../src/dcr-diagnostics.ts';
test('registration diagnostics omit tokens, secrets, names and callback queries',()=>{
 const result=registrationDiagnostic({client_name:'private-name',client_secret:'sensitive-client-secret',redirect_uris:['https://backend.composio.dev/callback?state=private-state'],logo_uri:null,token_endpoint_auth_method:'client_secret_post',grant_types:['authorization_code','refresh_token']},400,{error:'invalid_client_metadata',error_description:'Invalid logo_uri: expected string, got object',access_token:'secret-access'});
 const serialized=JSON.stringify(result);assert.doesNotMatch(serialized,/private-name|sensitive-client-secret|private-state|secret-access/);assert.equal(result.invalidField,'logo_uri');assert.equal(result.status,400);assert.deepEqual(result.redirects,[{scheme:'https:',host:'backend.composio.dev',userinfo:false,fragment:false}]);
});
test('untrusted descriptions and unknown errors are never retained',()=>{
 const result=registrationDiagnostic({token_endpoint_auth_method:'malicious secret'},500,{error:'secret-value',error_description:'secret-output'});
 assert.doesNotMatch(JSON.stringify(result),/malicious|secret-value|secret-output/);assert.equal(result.method,'unsupported');assert.equal(result.error,'unclassified');
});
