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
  TEXT_FALLBACK    （可选）纯文本 fallback，默认 "This email requires an HTML viewer."
"""

import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from email.utils import formatdate


def main() -> int:
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    mail_from = os.environ.get("MAIL_FROM") or smtp_user
    mail_to_raw = os.environ.get("MAIL_TO", "")
    mail_subject = os.environ.get("MAIL_SUBJECT", "eess.IV Daily")
    html_path = os.environ.get("HTML_PATH", "docs/email-body.html")
    text_fallback = os.environ.get(
        "TEXT_FALLBACK",
        "本邮件包含 HTML 内容，请使用支持 HTML 的邮件客户端查看。"
    )

    missing = [k for k, v in {
        "SMTP_HOST": smtp_host,
        "SMTP_USER": smtp_user,
        "SMTP_PASS": smtp_pass,
        "MAIL_TO": mail_to_raw,
    }.items() if not v]
    if missing:
        print(f"[send_email] missing env vars: {missing}", file=sys.stderr)
        return 2

    recipients = [t.strip() for t in mail_to_raw.split(",") if t.strip()]
    if not recipients:
        print("[send_email] MAIL_TO has no valid recipients", file=sys.stderr)
        return 2

    if not os.path.exists(html_path):
        print(f"[send_email] HTML body not found: {html_path}", file=sys.stderr)
        return 2

    with open(html_path, "r", encoding="utf-8") as f:
        html_body = f.read()

    msg = EmailMessage()
    msg["From"] = mail_from
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = mail_subject
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(text_fallback)
    msg.add_alternative(html_body, subtype="html")

    print(f"[send_email] connecting to {smtp_host}:{smtp_port} as {smtp_user}")
    print(f"[send_email] recipients: {recipients}")
    print(f"[send_email] subject: {mail_subject}")
    print(f"[send_email] html body: {html_path} ({len(html_body)} bytes)")

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=ctx, timeout=20) as server:
            server.login(smtp_user, smtp_pass)
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


if __name__ == "__main__":
    sys.exit(main())
