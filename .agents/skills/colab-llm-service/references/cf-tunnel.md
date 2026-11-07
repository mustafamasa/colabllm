# Cloudflare backend and global domains

Read the current configuration with `colabllm tunnel-config`. The shared file is
`~/.colabllm/tunnel-config.json`; changing it does not move existing runtime connections.
FRP enabled there makes FRP the default. To operate Cloudflare for one launch, pass
`--transport cloudflare`, or `--quick-tunnel` for a dynamic URL regardless of saved domains.

Domain precedence: explicit `--domain HOST`, one-launch `--base-domain`,
`COLABSVC_BASE_DOMAIN`, saved project base domain, then a dynamic URL. The three hostname
options (`--domain`, `--base-domain`, `--quick-tunnel`) are mutually exclusive.
Service aliases and instance suffixes are retained, e.g. `qwen-2.example.com`.
`Service.zone` is a historical alias suffix, not a default deployment target.

Global configuration commands are documented in
[README.md](../../../../README.md#cloudflare-tunnel-yapılandırması).
Use `colabllm tunnel-config --clear` to clear the saved domain; if FRP is enabled,
disable it first because FRP requires a hostname. Environment overrides still apply.

## Named tunnel versus Quick Tunnel

With a hostname, `src/colabllm/lib/colabsvc.py:ensure_named_tunnel()` uses the local `cf` login to
create/reuse `colab-<service-instance>`, configure ingress to `svc.tunnel_port`, set a
proxied CNAME, and retrieve the token. Zone lookup uses the longest matching suffix.
On Colab, `templates/tunnel_client.py` starts cloudflared; registration is logged in
`/content/ct/cloudflared.log`.

Without a hostname, cloudflared starts a Quick Tunnel and reports a random
`https://*.trycloudflare.com` URL. This path needs no `cf` login. Omitting `--domain`
does not imply a Quick Tunnel when a project domain or FRP backend is configured.

## DNS and lifecycle

- Use `cf auth whoami` and `cf zones list` to diagnose account/Zone access.
- Keep DNS ownership checks in the shared backend. Unrelated records and multiple
  conflicting records cause an error. Switching this instance's owned FRP record to a
  named tunnel is supported; generic A/AAAA records are not safe to overwrite.
- `cf`'s beta `--tunnel-config-*` flags can demand unrelated Access options. The runner
  uses `--body` JSON for ingress updates; retain that approach.
- Tokens live in private `~/.colabllm/services/` data and injected environment, not notebook source.
- `down` retains named tunnels/DNS. `down --delete-tunnel` cleans recorded resources;
  separate Cloudflare and FRP identities are preserved across backend switches.
  Use the recorded hostname/Zone, not the current global defaults, for cleanup.
