"""
send_email.py
- 通过 SMTP_SSL（隐式 TLS，端口 465）发送邮件
- 适配 163 邮箱要求
- 凭据从环境变量读取，绝不写进代码或 config

环境变量：
  SMTP_HOST        e.g. smtp.163.com
  SMTP_PORT        e.g. 465
  SMTP_USER        e.g. 13261281092@163.com
  SMTP_PASS        163 授权码（不是登录密码）
  MAIL_FROM        （可选）默认等于 SMTP_USER
  MAIL_TO          逗号分隔的收件人列表
  MAIL_SUBJECT     邮件主题
  HTML_PATH        HTML 正文文件路径
  TEXT_FALLBACK    （可选）纯文本文案

子命令：
  python send_email.py            # 实际发送
  python send_email.py --check    # 仅做诊断：DNS / TCP / AUTH 可达性，不发邮件
"""

import argparse
import os
import smtplib
import socket
import ssl
import sys
from email.message import EmailMessage
from email.utils import formatdate


def _load_env():
    smtp_host = (os.environ.get("SMTP_HOST") or "").strip()
    smtp_port_raw = (os.environ.get("SMTP_PORT") or "465").strip()
    smtp_user = (os.environ.get("SMTP_USER") or "").strip()
    smtp_pass = os.environ.get("SMTP_PASS")  # 密码禁止 strip（怕用户密码含空白符）
    mail_from = (os.environ.get("MAIL_FROM") or smtp_user).strip()
    mail_to_raw = os.environ.get("MAIL_TO", "")
    mail_subject = (os.environ.get("MAIL_SUBJECT") or "eess.IV Daily").strip()
    html_path = (os.environ.get("HTML_PATH") or "docs/email-body.html").strip()
    text_fallback = os.environ.get(
        "TEXT_FALLBACK",
        "本邮件包含 HTML 内容，请使用支持 HTML 的邮件客户端查看。"
    )
    try:
        smtp_port = int(smtp_port_raw)
    except ValueError:
        smtp_port = 465
    return {
        "smtp_host": smtp_host, "smtp_port": smtp_port,
        "smtp_user": smtp_user, "smtp_pass": smtp_pass,
        "mail_from": mail_from, "mail_to_raw": mail_to_raw,
        "mail_subject": mail_subject, "html_path": html_path,
        "text_fallback": text_fallback,
    }


def _check_missing(env):
    return [k for k, v in {
        "SMTP_HOST": env["smtp_host"],
        "SMTP_USER": env["smtp_user"],
        "SMTP_PASS": env["smtp_pass"],
        "MAIL_TO": env["mail_to_raw"],
    }.items() if not v]


def cmd_check(env) -> int:
    """诊断：DNS / TCP / AUTH 都不真正发邮件"""
    print("=" * 60)
    print("[check] Environment snapshot")
    print("=" * 60)
    # 打印原始值（包括 strip 前后的差异，方便发现 secret 中是否有不可见字符）
    for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS",
              "MAIL_FROM", "MAIL_TO", "MAIL_SUBJECT", "HTML_PATH"):
        raw = os.environ.get(k)
        if raw is None:
            print(f"  {k:14s} = (not set)")
        elif k == "SMTP_PASS":
            # 永远不打印密码原文/编码，只报告是否设置与长度
            print(f"  {k:14s} = (set, {len(raw)} chars)")
        else:
            stripped_diff = "" if raw == raw.strip() else f"  [stripped->{raw.strip()!r}]"
            print(f"  {k:14s} = {raw!r}{stripped_diff}")
    print()

    host = env["smtp_host"]
    port = env["smtp_port"]

    if not host:
        print("[check] FAIL: SMTP_HOST is empty")
        return 2

    # 1. DNS 解析
    print(f"[check] step 1: DNS resolve {host!r}")
    try:
        addrs = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        ips = sorted({a[4][0] for a in addrs})
        print(f"[check]   OK -> {ips}")
    except socket.gaierror as e:
        print(f"[check]   FAIL: {e}")
        print(f"[check]   HINT: check SMTP_HOST secret value (常见错误：拼写错 / 带 https:// / 带 :端口)")
        return 4

    # 2. TCP 连接
    print(f"[check] step 2: TCP connect {host}:{port}")
    try:
        with socket.create_connection((host, port), timeout=10) as s:
            print(f"[check]   OK (local={s.getsockname()}, peer={s.getpeername()})")
    except Exception as e:
        print(f"[check]   FAIL: {e}")
        print(f"[check]   HINT: 163 应使用 465 隐式 SSL；如用 STARTTLS 应改 587")
        return 5

    # 3. SMTP SSL 握手 + 认证
    print(f"[check] step 3: SMTP_SSL handshake + AUTH")
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=20) as server:
            server.ehlo()
            print(f"[check]   handshake OK")
            server.login(env["smtp_user"], env["smtp_pass"])
            print(f"[check]   AUTH OK as {env['smtp_user']}")
    except smtplib.SMTPAuthenticationError as e:
        print(f"[check]   FAIL: auth error {e}")
        print(f"[check]   HINT: 163 需要 16 位授权码，不是登录密码")
        return 3
    except Exception as e:
        print(f"[check]   FAIL: {e}")
        return 1

    print()
    print("[check] all steps passed; ready to send")
    return 0


def cmd_send(env) -> int:
    missing = _check_missing(env)
    if missing:
        print(f"[send_email] missing env vars: {missing}", file=sys.stderr)
        return 2

    recipients = [t.strip() for t in env["mail_to_raw"].split(",") if t.strip()]
    if not recipients:
        print("[send_email] MAIL_TO has no valid recipients", file=sys.stderr)
        return 2

    if not os.path.exists(env["html_path"]):
        print(f"[send_email] HTML body not found: {env['html_path']}", file=sys.stderr)
        return 2

    with open(env["html_path"], "r", encoding="utf-8") as f:
        html_body = f.read()

    msg = EmailMessage()
    msg["From"] = env["mail_from"]
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = env["mail_subject"]
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(env["text_fallback"])
    msg.add_alternative(html_body, subtype="html")

    print(f"[send_email] connecting to {env['smtp_host']}:{env['smtp_port']} as {env['smtp_user']}")
    print(f"[send_email] recipients: {recipients}")
    print(f"[send_email] subject: {env['mail_subject']}")
    print(f"[send_email] html body: {env['html_path']} ({len(html_body)} bytes)")

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(env["smtp_host"], env["smtp_port"], context=ctx, timeout=20) as server:
            server.login(env["smtp_user"], env["smtp_pass"])
            refused = server.send_message(msg)
            if refused:
                print(f"[send_email] some recipients refused: {refused}", file=sys.stderr)
                return 1
        print("[send_email] sent OK")
        return 0
    except smtplib.SMTPAuthenticationError as e:
        print(f"[send_email] auth failed (check 163 authorization code): {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[send_email] error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="only diagnose DNS/TCP/AUTH, do not send")
    args = parser.parse_args()

    env = _load_env()
    if args.check:
        return cmd_check(env)
    return cmd_send(env)


if __name__ == "__main__":
    sys.exit(main())
