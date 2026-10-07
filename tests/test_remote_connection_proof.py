"""Native device signing uses maintained cryptography/PyJWT primitives."""
import hashlib
import json
import time
import jwt
import pytest
try:
    from superlocalmemory.remote_connections.proof import DeviceSigner
except ImportError:
    DeviceSigner=None

def test_device_key_is_private_and_public_thumbprint_is_stable():
    assert DeviceSigner is not None
    signer=DeviceSigner.generate()
    assert 'PRIVATE KEY' not in repr(signer)
    assert set(signer.public_jwk)=={'kty','crv','x','y'} and signer.public_jwk['crv']=='P-256'
    assert 'd' not in signer.public_jwk
    restored=DeviceSigner(signer.private_pem)
    assert restored.thumbprint==signer.thumbprint
    assert len(signer.thumbprint)==43

def test_signed_proof_binds_token_method_target_and_fresh_nonce():
    assert DeviceSigner is not None
    signer=DeviceSigner.generate()
    proof=signer.proof('GET','https://connect.superlocalmemory.com/connector',token='synthetic-token')
    public=jwt.PyJWK.from_json(json.dumps(signer.public_jwk)).key
    claims=jwt.decode(proof,public,algorithms=['ES256'],options={'verify_aud':False})
    assert claims['htm']=='GET' and claims['htu']=='https://connect.superlocalmemory.com/connector'
    assert abs(claims['iat']-time.time())<5 and 'synthetic-token' not in proof
    other=jwt.decode(signer.proof('GET','https://connect.superlocalmemory.com/connector',token='synthetic-token'),public,algorithms=['ES256'])
    assert claims['jti']!=other['jti']

@pytest.mark.parametrize('key',['bad','-----BEGIN PRIVATE KEY-----\nbad\n-----END PRIVATE KEY-----'])
def test_invalid_private_key_is_sanitized(key):
    assert DeviceSigner is not None
    with pytest.raises(ValueError,match='invalid_device_key'):DeviceSigner(key)

def test_proof_refuses_insecure_or_ambiguous_target():
    assert DeviceSigner is not None
    signer=DeviceSigner.generate()
    for target in ['http://connect.superlocalmemory.com/connector','https://name:pass@connect.superlocalmemory.com/connector','https://connect.superlocalmemory.com/connector#fragment']:
        with pytest.raises(ValueError):signer.proof('GET',target,token='synthetic')
