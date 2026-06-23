<p align="center">
  <h1 align="center"><br><ins>eess.IV-ARXIV-DAILY</ins><br>Automatically Update arXiv eess.IV Papers Daily via GitHub Actions</h1>
</p>

## Overview

This repository crawls the arXiv **eess.IV** (Image and Video Processing) category
every day, drops medical papers, extracts GitHub repository links, translates
abstracts to Chinese (via the free MyMemory API), then publishes the result to:

- `README.md` at the repo root
- a daily email (163 SMTP) to subscribers
- (optional) a Feishu webhook card

Repository layout:

- `daily_arxiv.py` — main script
- `config.yaml` — category, medical filter, translation, email config
- `scripts/build_email.py` — render the JSON digest to an HTML email body
- `scripts/send_email.py` — send the HTML email via SMTP_SSL (port 465)
- `.github/workflows/cv-arxiv-daily.yml` — daily cron + email steps

## Usage

<details>
  <summary>Table of Contents</summary>

1. Fork this repo.
2. Edit `.github/workflows/cv-arxiv-daily.yml` and `config.yaml` as needed; push to remote.
3. Enable GitHub Actions:
   - Setting → Actions → Workflow permissions → **Read and write permissions** → save.
   - Actions → enable the `Run eess.IV Daily` workflow. Click **Run workflow** once
     to verify the first run succeeds (it will email you the same digest the
     daily cron will deliver).
4. Configure the 6 required GitHub Secrets (see **邮件订阅** below).
5. (Optional) Edit `medical_exclude_keywords` in `config.yaml` to add domain
   keywords you want to filter out.
6. (Optional) For history mining, manually run the `Update History Papers`
   workflow with `history_date_from` / `history_date_to` / `min_citations`
   inputs. It produces `docs/eess-iv-history.json`.

</details>

## 邮件订阅

GitHub Actions sends the daily digest via 163 SMTP. The workflow reads the
following **Repository Secrets** (Settings → Secrets and variables → Actions
→ New repository secret):

| Secret | Example value | Notes |
|---|---|---|
| `SMTP_HOST` | `smtp.163.com` | 163 邮箱的 SMTP 服务器 |
| `SMTP_PORT` | `465` | 隐式 SSL 端口 |
| `SMTP_USER` | `13261281092@163.com` | 163 邮箱地址 |
| `SMTP_PASS` | `<16 位授权码>` | **授权码**，不是登录密码。在 163 邮箱设置 → POP3/SMTP/IMAP 中开启并生成 |
| `MAIL_TO_1` | `13261281092@163.com` | 第一个收件人 |
| `MAIL_TO_2` | `duanchenhui.zoro@jd.com` | 第二个收件人（163 会代为转发） |
| `FEISHU_WEBHOOK` | `https://open.feishu.cn/...` | （可选）飞书机器人 webhook |

### 163 授权码获取

1. 登录 163 邮箱网页版
2. 顶部 **设置** → **POP3/SMTP/IMAP**
3. 开启 **SMTP 服务** 和 **IMAP/SMTP 服务**（如果提示）
4. 扫码验证后系统会生成 16 位 **授权码**（注意保存，只显示一次）
5. 把授权码填到 `SMTP_PASS` Secret

> 第一次发邮件大概率进入收件人的**垃圾邮件**，请在邮件客户端里把发件人加白名单。

## 手动触发

`Run eess.IV Daily` workflow 支持 `workflow_dispatch` 手动触发。手动触发与每日定时触发的
行为完全一致：抓取**当天**（Asia/Shanghai 时区）的 eess.IV 新增论文。日期由工作流根据
Asia/Shanghai 时区自动计算，无法覆盖。

## 翻译说明

- 用 MyMemory 免费 API（`en|zh-CN`），每日 5000 字符/IP 限制
- `config.yaml` 中 `translation.max_translate_per_day: 10` 控制每天翻译前 N 篇摘要
- 超出限制或失败的论文会自动回退到 240 字截断的英文摘要

## 历史高引论文

- `update-history.yml` 手动触发，输入日期范围与最低引用数
- 输出 `docs/eess-iv-history.json`（按引用数降序）
- 抓取范围固定为 `cat:eess.IV`，可修改 `config.yaml.history_date_from/to`
