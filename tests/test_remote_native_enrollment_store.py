import pytest
from superlocalmemory.remote_connections.native_enrollment import PendingEnrollment, NativeEnrollmentStore
from superlocalmemory.remote_connections.proof import DeviceSigner

class Backend:
    def __init__(self): self.values={}
    def get_password(self, service, name): return self.values.get((service,name))
    def set_password(self, service, name, value): self.values[(service,name)]=value

def record():
    return PendingEnrollment(installation_id='installation', owner='owner', profile='profile', connection_id='a'*32, redirect_uri='http://127.0.0.1:18767/api/v3/connections/callback', state='s'*43, verifier='v'*43, private_key=DeviceSigner.generate().private_pem, intent_json='{}', expires_at_ms=9999999999999)

def test_pending_material_never_appears_in_repr():
    row=record()
    assert row.verifier not in repr(row)
    assert 'PRIVATE KEY' not in repr(row)

def test_pending_state_is_bound_to_stored_connection_and_cancelled_terminally(tmp_path):
    store=NativeEnrollmentStore(tmp_path, backend=Backend())
    row=record();store.save(row)
    assert store.by_state(row.state)==row
    store.cancel(row.connection_id)
    assert store.by_state(row.state) is None
    with pytest.raises(ValueError,match='enrollment_cancelled'):store.save(row)

def test_expired_completed_record_remains_available_only_for_authenticated_cleanup(tmp_path):
    from dataclasses import replace
    backend=Backend();store=NativeEnrollmentStore(tmp_path,backend=backend,clock=lambda:100)
    row=replace(record(),expires_at_ms=101000,completed=True)
    store.save(row)
    later=NativeEnrollmentStore(tmp_path,backend=backend,clock=lambda:102)
    assert later.by_connection(row.connection_id)==row
    assert later.by_state(row.state) is None
    later.cancel(row.connection_id)
    assert later.by_connection(row.connection_id) is None

def test_expired_pending_has_cleanup_only_lookup_without_callback_revival(tmp_path):
    from dataclasses import replace
    backend=Backend();store=NativeEnrollmentStore(tmp_path,backend=backend,clock=lambda:100)
    row=replace(record(),expires_at_ms=101000)
    store.save(row)
    later=NativeEnrollmentStore(tmp_path,backend=backend,clock=lambda:102)
    assert later.by_connection(row.connection_id) is None
    assert later.by_state(row.state) is None
    assert later.by_connection(row.connection_id,for_cleanup=True)==row
    later.cancel(row.connection_id)
    assert later.by_connection(row.connection_id,for_cleanup=True) is None
