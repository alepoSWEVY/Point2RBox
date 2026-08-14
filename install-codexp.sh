#!/usr/bin/env bash
set -euo pipefail

install_dir="${HOME}/.local/bin"
target="${install_dir}/codexp"
remote_port="${1:-7890}"
host_name="$(hostname -s)"
config_dir="${HOME}/.config/codexp"
port_file="${config_dir}/proxy-port-${host_name}"

case "${remote_port}" in
  ''|*[!0-9]*)
    echo "[ERROR] Proxy port must be a number." >&2
    exit 1
    ;;
esac

mkdir -p "${install_dir}"
mkdir -p "${config_dir}"
printf '%s\n' "${remote_port}" >"${port_file}"

cat >"${target}" <<'CODEXP'
#!/usr/bin/env bash
set -euo pipefail

host_name="$(hostname -s)"
port_file="${HOME}/.config/codexp/proxy-port-${host_name}"
remote_port="7890"

if [[ -r "${port_file}" ]]; then
  IFS= read -r configured_port <"${port_file}" || true
  case "${configured_port:-}" in
    ''|*[!0-9]*) ;;
    *) remote_port="${configured_port}" ;;
  esac
fi

proxy_url="http://127.0.0.1:${remote_port}"

if ! command -v codex >/dev/null 2>&1; then
  echo "[ERROR] codex was not found in PATH." >&2
  exit 1
fi

echo "[codexp] Testing SSH proxy at ${proxy_url} ..."

if command -v curl >/dev/null 2>&1; then
  # Run the full test with the same temporary proxy environment used by Codex.
  http_code="$({
    HTTP_PROXY="${proxy_url}" \
    HTTPS_PROXY="${proxy_url}" \
    http_proxy="${proxy_url}" \
    https_proxy="${proxy_url}" \
    NO_PROXY="localhost,127.0.0.1" \
    no_proxy="localhost,127.0.0.1" \
    curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
      --connect-timeout 8 --max-time 15 https://api.openai.com/v1/models
  } || true)"

  case "${http_code}" in
    200|401)
      echo "[codexp] Proxy test passed (OpenAI HTTP ${http_code})."
      ;;
    403)
      echo "[ERROR] OpenAI returned HTTP 403. Check the region of your Windows proxy exit node." >&2
      exit 1
      ;;
    000|"")
      echo "[ERROR] Cannot reach OpenAI through 127.0.0.1:${remote_port}." >&2
      echo "        Start the Windows tunnel first and keep it open." >&2
      exit 1
      ;;
    *)
      echo "[ERROR] Proxy test failed (OpenAI HTTP ${http_code})." >&2
      exit 1
      ;;
  esac
else
  # Minimal tunnel check for servers where curl is unavailable.
  if (exec 3<>/dev/tcp/127.0.0.1/"${remote_port}") 2>/dev/null; then
    exec 3>&-
    exec 3<&-
    echo "[codexp] SSH proxy port is reachable (curl unavailable; full test skipped)."
  else
    echo "[ERROR] Cannot connect to 127.0.0.1:${remote_port}." >&2
    echo "        Start the Windows tunnel first and keep it open." >&2
    exit 1
  fi
fi

echo "[codexp] Starting Codex with temporary proxy variables ..."
exec env \
  HTTP_PROXY="${proxy_url}" \
  HTTPS_PROXY="${proxy_url}" \
  http_proxy="${proxy_url}" \
  https_proxy="${proxy_url}" \
  NO_PROXY="localhost,127.0.0.1" \
  no_proxy="localhost,127.0.0.1" \
  codex "$@"
CODEXP

chmod 755 "${target}"

case ":${PATH}:" in
  *":${install_dir}:"*) ;;
  *)
    shell_rc="${HOME}/.bashrc"
    path_line='export PATH="$HOME/.local/bin:$PATH"'
    if ! grep -Fqx "${path_line}" "${shell_rc}" 2>/dev/null; then
      printf '\n%s\n' "${path_line}" >>"${shell_rc}"
    fi
    export PATH="${install_dir}:${PATH}"
    echo "Added ${install_dir} to PATH in ${shell_rc}."
    ;;
esac

echo "Installed: ${target}"
echo "Configured proxy port for ${host_name}: ${remote_port}"
echo "Run this now:"
echo "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
echo "  codexp"
