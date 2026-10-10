# Company mode

By default SuperLocalMemory is single-user: whoever runs it owns it, and there is
no login. Company mode turns that off and requires every operation on your
memories to be attributed to a named person with a role on that workspace.

Turn it on only when more than one person shares a store. It adds a login step to
everything, including the connection your AI assistant uses.

## Roles

| Role | Read | Write | Delete | Share | Manage users |
|---|---|---|---|---|---|
| `admin` | yes | yes | yes | yes | yes |
| `member` | yes | yes | — | yes | — |
| `viewer` | yes | — | — | — | — |

Roles are granted per workspace. A role in one workspace grants nothing in
another — someone with a valid login and no role here has no access here.

While no user is enrolled, the machine operator can still create the first
administrator. Once users exist and login is required, the install token or an
API key alone no longer administers the workspace: any program on the computer
can fetch the install token, so it would let anyone switch company mode off.
Administration then needs a signed-in admin, or `slm team` run on the SLM
computer by the user who runs SLM (see below), which cannot be reached over the
network. What company mode removes is the operator's *unattributed access to
data*.

## Setting it up

Use the dashboard: **Settings → Access**. Add users, grant each a role on the
workspace, then turn on "Require login".

There is no `slm user`, `slm role` or `slm company-mode`. Users and roles are
managed in the dashboard or over the HTTP API below. The one CLI is `slm team`,
for the login requirement only:

```bash
slm team status                       # Require login: on|off. Users: N.
slm team policy --require-login on    # every user signs in
slm team policy --require-login off   # the machine owner is the user again
```

`slm team` talks to the running daemon with a private capability file that only
the user running SLM can read, so the owner can always switch the requirement
off, even with every administrator locked out. Run it as that user. If the
daemon refuses it, it says to run it as the same user, with the daemon running
from that user's data folder.

The same operations over HTTP, for scripting. Every one of these also needs
machine authentication — see [auth-write-gate.md](auth-write-gate.md) for the
header your client must send; the dashboard supplies it for you.

```
POST   /api/rbac/users     {"username": ..., "password": ..., "display_name": ""}
POST   /api/rbac/members   {"user_id": ..., "role": "viewer"}
POST   /api/rbac/policy    {"require_login": true}
GET    /api/rbac/status
```

`POST /api/rbac/members` grants a role on the **active** workspace — it takes no
workspace argument, so switch to the right one first.

`GET /api/rbac/status` is how you confirm it took effect. It reports
`require_login`, `rbac_active`, and `user_count`. Check it before you rely on the
mode being on.

## Logging in

`POST /api/rbac/login` with a username and password. A browser receives an
HttpOnly cookie and nothing else — the token is never in the response body, so it
cannot be picked out of proxy logs or devtools. A non-browser client receives the
token in the body, because it has no cookie jar to use.

Sessions last 12 hours. Expired ones are cleared when the service starts and
periodically thereafter.

## AI tools and other non-browser clients

This is the part that surprises people, so it is worth stating plainly.

Your AI assistant does not connect through the dashboard. It uses a separate
connection with no browser and no cookie, so in company mode it cannot say who is
calling — and **its writes are refused**. Reads are unaffected.

To let a tool write in company mode, give its process a session token:

```bash
export SLM_USER_SESSION="<token from POST /api/rbac/login>"
```

Start the tool in that environment. The write is then attributed to that user and
checked against that user's role on the workspace, exactly as through the
dashboard. A `viewer` still cannot write, whatever the tool asks for.

If the variable is unset, expired, or not a real session, the write is refused
rather than treated as the machine owner's. That is deliberate: a channel that
cannot prove who is calling must not act as the owner. It is also the defect this
behaviour replaced — the setting used to be read by the web interface and not by
the tool connection, so per-user access applied to one and not the other.

## What login covers

With login required, every page of data needs a signed-in user: recall and
saving, pictures, documents and connected folders, export, compliance records,
profiles, learning and trust details. A test walks every route so a new one
cannot be added without the check. Reading a file by path, connecting a folder
and turning features on (pictures and documents) need the owner or an admin, and
SLM's own data folder can never be named. A connected web app cannot change,
delete, pin or replace a memory it is not allowed to see; it is answered as if
the memory did not exist.

In strict mode (`SLM_REQUIRE_CREDENTIALS=1`) the dashboard is not handed the SLM
key; print it with `slm token show` and paste it once. See
[auth-write-gate.md](auth-write-gate.md).

## What company mode does not do

- **It is not encryption.** Anyone who can read the database file can read your
  memories. Company mode governs access *through* SuperLocalMemory, not access to
  the disk. For that, see
  [SECURITY-encryption-at-rest.md](SECURITY-encryption-at-rest.md).
- **It does not partition a workspace between users.** Everyone with a role on a
  workspace sees all of it. Separate workspaces are the unit of separation.
- **It does not reach a second machine.** Users and roles live in the store, so
  another machine with its own store has its own users.

## Turning it off

Set `require_login` back to false, from the dashboard as an admin or with
`slm team policy --require-login off` on the SLM computer. Users, roles and
sessions are kept; reads and writes stop requiring a login, and the operator's
unattributed access returns.
