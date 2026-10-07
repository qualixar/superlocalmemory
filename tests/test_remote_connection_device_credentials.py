from superlocalmemory.remote_connections.credentials import ConnectorCredential
from superlocalmemory.remote_connections.proof import DeviceSigner

def test_device_private_key_is_protected_and_available_for_companion():
    signer = DeviceSigner.generate()
    value = ConnectorCredential(installation_id='installation', owner='owner', profile='profile', connection_id='a'*32, generation=1, expires_at_ms=9999999999999, device_token='b'*64, origin_key='slmr_'+'c'*43, device_private_key=signer.private_pem)
    assert value.device_private_key == signer.private_pem
    assert 'PRIVATE KEY' not in repr(value)
