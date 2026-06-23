"""
eess.IV Daily Arxiv
- 抓取 arXiv eess.IV (Image and Video Processing) 类别每日新增论文
- 过滤医学领域论文
- 提取 GitHub 仓库链接
- 调用 MyMemory 免费 API 翻译摘要为中文
- 渲染 README.md；可选推送飞书 + 邮件
"""

import os
import re
import json
import time
import html
import arxiv
import yaml
import logging
import argparse
import datetime
import requests

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

arxiv_url = "http://arxiv.org/"
semantic_scholar_url = "https://api.semanticscholar.org/graph/v1/paper/arXiv:"

# GitHub 链接抽取正则与尾部剥离字符
GITHUB_RE = re.compile(
    r'https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]*)?'
)
STRIP_CHARS = '.,;:!?)>]\"'

# 模块级翻译计数（避免超额调用 MyMemory 限额）
_translate_count_today = 0


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------

def load_config(config_file: str) -> dict:
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logging.info(f'config loaded from {config_file}')
    return config


def get_authors(authors, first_author: bool = False) -> str:
    if first_author:
        return str(authors[0])
    return ", ".join(str(author) for author in authors)


def sort_papers(papers):
    """按 paper id 字符串倒序排（id 含版本号时也会自然反映更新时间）"""
    output = dict()
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output


def is_medical_paper(title: str, abstract: str, exclude_keywords: list) -> bool:
    """医学论文过滤：title + abstract 中命中任一关键词则丢弃"""
    text = f"{title} {abstract}".lower()
    return any(kw.lower() in text for kw in exclude_keywords)


def extract_github_url(*sources: str) -> str:
    """从 abstract / comment 字段中抽取第一个 github.com 链接，去尾部标点"""
    for src in sources:
        if not src:
            continue
        m = GITHUB_RE.search(src)
        if m:
            return m.group(0).rstrip(STRIP_CHARS)
    return ""


def translate_to_chinese(text: str, cfg: dict) -> str:
    """调用 MyMemory API 将英文翻译为简体中文；失败时按配置回退到原文"""
    global _translate_count_today

    if not cfg.get("enabled", False) or not text:
        return text

    if _translate_count_today >= cfg.get("max_translate_per_day", 10):
        return text if cfg.get("fallback_to_english", True) else ""

    endpoint = cfg.get("endpoint", "https://api.mymemory.translated.net/get")
    langpair = cfg.get("langpair", "en|zh-CN")
    timeout = cfg.get("timeout_sec", 5)
    max_chars = cfg.get("max_input_chars", 500)
    retries = cfg.get("retry", 2)
    fallback = cfg.get("fallback_to_english", True)

    # 截断过长输入
    src = text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "…"

    for attempt in range(retries + 1):
        try:
            r = requests.get(endpoint, params={"q": src, "langpair": langpair}, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                translated = data.get("responseData", {}).get("translatedText", "")
                # MyMemory 在配额耗尽时会返回以 "MYMEMORY WARNING" 开头的内容
                if translated and "MYMEMORY WARNING" not in translated.upper()[:30]:
                    _translate_count_today += 1
                    return html.unescape(translated)
            logging.warning(f"Translate attempt {attempt + 1} bad status={r.status_code}")
        except Exception as e:
            logging.warning(f"Translate attempt {attempt + 1} failed: {e}")
        time.sleep(1 + attempt)

    return text if fallback else ""


# ----------------------------------------------------------------------
# 引用统计（仅历史模式使用）
# ----------------------------------------------------------------------

def get_paper_citations(arxiv_id: str, retry: int = 3) -> int:
    arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
    url = f"{semantic_scholar_url}{arxiv_id}?fields=citationCount"
    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.json().get('citationCount', 0)
            elif response.status_code == 404:
                return 0
            else:
                time.sleep(2)
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2)
            else:
                logging.warning(f"Failed to get citations for {arxiv_id}: {e}")
                return 0
    return 0


def get_papers_citations_batch(arxiv_ids: list, retry: int = 3) -> dict:
    """批量调用 Semantic Scholar batch 接口取引用数"""
    if not arxiv_ids:
        return {}

    clean_ids = [re.sub(r'v\d+$', '', aid) for aid in arxiv_ids]
    prefixed_ids = [f"ARXIV:{aid.upper()}" for aid in clean_ids]
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params = {"fields": "paperId,citationCount,externalIds"}
    data = {"ids": prefixed_ids}

    for attempt in range(retry):
        try:
            response = requests.post(url, json=data, params=params, timeout=60)
            if response.status_code == 200:
                results = response.json()
                citations_dict = {}
                for item in results:
                    if item and item.get('externalIds'):
                        ext_ids = item.get('externalIds', {})
                        arxiv_id = ext_ids.get('ArXiv', '')
                        citations_dict[arxiv_id.upper()] = item.get('citationCount', 0)
                return citations_dict
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 30
                logging.warning(f"Batch rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.warning(f"Batch request failed: {response.status_code}, response: {response.text[:200]}")
                time.sleep(5)
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(5)
            else:
                logging.warning(f"Batch citations failed: {e}")
                return {}
    return {}


# ----------------------------------------------------------------------
# arXiv 抓取
# ----------------------------------------------------------------------

def search_arxiv_with_retry(category: str, date_from: str, date_to: str,
                            max_results: int = 300, max_retries: int = 3) -> list:
    """按 arxiv 类目 + 提交日期区间搜索（arxiv>=2.0 用 Client）"""
    date_filter = f"submittedDate:[{date_from}0000 TO {date_to}2359]"
    query = f"cat:{category} AND {date_filter}"
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)

    for attempt in range(max_retries):
        try:
            search_engine = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            return list(client.results(search_engine))
        except Exception as e:
            error_msg = str(e)
            if "Rate exceeded" in error_msg or "429" in error_msg:
                wait_time = (attempt + 1) * 30
                logging.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logging.error(f"Arxiv search error: {e}")
                if attempt == max_retries - 1:
                    return []
                time.sleep(10)
    return []


def get_daily_papers(category: str, max_results: int,
                     date_from: str, date_to: str,
                     exclude_keywords: list,
                     github_cfg: dict,
                     translation_cfg: dict) -> tuple:
    """
    拉取指定日期的 eess.IV 新论文。
    返回 (data, data_web) — 两者结构相同，便于兼容后续输出渲染。
    """
    time.sleep(3)

    target_day = datetime.date(
        int(date_from[:4]), int(date_from[4:6]), int(date_from[6:8])
    )
    results = search_arxiv_with_retry(category, date_from, date_to, max_results)

    content = {}
    content_web = {}
    skipped_medical = 0
    skipped_date = 0

    for r in results:
        # arxiv 跨时区公告：按 published 日期二次过滤
        if r.published.date() != target_day:
            skipped_date += 1
            continue

        title = r.title
        abstract = r.summary.replace("\n", " ")

        if is_medical_paper(title, abstract, exclude_keywords):
            logging.info(f"[skip medical] {title[:60]}")
            skipped_medical += 1
            continue

        short_id = r.get_short_id()
        ver_pos = short_id.find('v')
        arxiv_id = short_id[:ver_pos] if ver_pos != -1 else short_id
        arxiv_abs_url = f"{arxiv_url}abs/{arxiv_id}"
        arxiv_pdf_url = f"{arxiv_url}pdf/{arxiv_id}"

        # GitHub 链接扫描 abstract + comment 字段
        comment = getattr(r, "comment", "") or ""
        code_url = extract_github_url(abstract, comment)

        # 中文摘要（带每日限额）
        summary_zh = translate_to_chinese(abstract, translation_cfg)

        first_author = str(r.authors[0]) if r.authors else "Unknown"
        authors = ", ".join(str(a) for a in r.authors)

        content[arxiv_id] = {
            "update_time": str(r.updated.date()),
            "title": title,
            "first_author": first_author,
            "authors": authors,
            "url": arxiv_abs_url,
            "pdf_url": arxiv_pdf_url,
            "code_url": code_url,
            "category": r.primary_category,
            "summary_en": abstract,
            "summary_zh": summary_zh,
        }
        content_web[arxiv_id] = content[arxiv_id]
        logging.info(f"[keep] {title[:60]} | code={bool(code_url)} | zh={bool(summary_zh)}")

    logging.info(
        f"Daily scan done: {len(content)} kept, "
        f"{skipped_medical} medical skipped, {skipped_date} date-mismatched"
    )
    return {category: content}, {category: content_web}


# ----------------------------------------------------------------------
# 历史高引论文（手动触发）
# ----------------------------------------------------------------------

def get_history_papers(category: str, query: str, max_results: int = 100,
                        date_from: str = "20100101", date_to: str = None,
                        min_citations: int = 0) -> dict:
    content = {}

    if not date_to:
        date_to = datetime.date.today().strftime('%Y%m%d')
    date_filter = f"submittedDate:[{date_from} TO {date_to}]"
    full_query = f"({query}) AND {date_filter}"

    search_max = 1000 if min_citations > 0 else max_results
    logging.info(f"History search: {date_from} to {date_to}, min_citations={min_citations}, search_max={search_max}")

    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search_engine = None
    for attempt in range(3):
        try:
            search_engine = arxiv.Search(
                query=full_query,
                max_results=search_max,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            break
        except Exception as e:
            error_msg = str(e)
            if "Rate exceeded" in error_msg or "429" in error_msg:
                wait_time = (attempt + 1) * 30
                logging.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logging.error(f"Arxiv search error: {e}")
                return {category: {}}

    paper_results = []
    for result in client.results(search_engine):
        paper_id = result.get_short_id()
        ver_pos = paper_id.find('v')
        paper_key = paper_id[:ver_pos] if ver_pos != -1 else paper_id
        paper_results.append((result, paper_key))

    # 批量取引用
    paper_keys = [pk for _, pk in paper_results]
    citations_dict = {}
    batch_size = 100
    for i in range(0, len(paper_keys), batch_size):
        batch = paper_keys[i:i + batch_size]
        logging.info(f"Fetching batch {i // batch_size + 1}/{(len(paper_keys) - 1) // batch_size + 1}...")
        batch_result = get_papers_citations_batch(batch)
        citations_dict.update(batch_result)
        time.sleep(1)

    papers_with_citations = []
    for result, paper_key in paper_results:
        citation_count = citations_dict.get(paper_key, 0)
        papers_with_citations.append({
            'result': result,
            'paper_key': paper_key,
            'citations': citation_count,
        })

    papers_with_citations.sort(key=lambda x: x['citations'], reverse=True)

    if min_citations > 0:
        papers_with_citations = [
            p for p in papers_with_citations
            if p['citations'] >= min_citations and p['citations'] > 0
        ]
        top_papers = papers_with_citations
    else:
        papers_with_citations = [p for p in papers_with_citations if p['citations'] > 0]
        top_papers = papers_with_citations[:max_results]

    for item in top_papers:
        result = item['result']
        paper_key = item['paper_key']
        citation_count = item['citations']
        paper_url = f"{arxiv_url}abs/{paper_key}"
        content[paper_key] = {
            "title": result.title,
            "authors": get_authors(result.authors),
            "first_author": get_authors(result.authors, first_author=True),
            "url": paper_url,
            "citations": citation_count,
            "published": result.published.date().isoformat(),
        }
    return {category: content}


def update_history_json(filename: str, data_dict: dict):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(filename, "r") as f:
        content = f.read()
        existing_data = json.loads(content) if content else {}

    paper_topics = {}
    for topic in existing_data:
        for paper_id in existing_data[topic]:
            paper_topics[paper_id] = topic

    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]
            if keyword not in existing_data:
                existing_data[keyword] = {}
            for paper_id, paper_info in papers.items():
                if paper_id in paper_topics and paper_topics[paper_id] != keyword:
                    continue
                if paper_id in existing_data[keyword]:
                    existing_citations = existing_data[keyword][paper_id].get('citations', 0)
                    if paper_info['citations'] > existing_citations:
                        existing_data[keyword][paper_id] = paper_info
                        paper_topics[paper_id] = keyword
                else:
                    existing_data[keyword][paper_id] = paper_info
                    paper_topics[paper_id] = keyword

    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    logging.info(f"History updated: {filename}")


# ----------------------------------------------------------------------
# 飞书推送
# ----------------------------------------------------------------------

def send_no_papers_message(webhook_url: str):
    try:
        data = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {"title": {"content": "📄 eess.IV Daily"}, "template": "blue"},
                "elements": [{"tag": "div", "text": {"tag": "lark_md", "content": "**今天没有新增 eess.IV 论文！**"}}],
            },
        }
        r = requests.post(webhook_url, json=data, timeout=10)
        logging.info(f"Feishu 'no papers' status: {r.status_code}")
    except Exception as e:
        logging.warning(f"Failed to send Feishu 'no papers' message: {e}")


def send_to_feishu(webhook_url: str, table_data: list):
    if not webhook_url or not table_data or len(table_data) < 2:
        return
    try:
        first_date = str(table_data[1][0]) if len(table_data) > 1 else ""
        paper_count = len(table_data) - 1

        elements = [{
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"📅 **{first_date}**  |  📚 **{paper_count}** 篇论文"},
        }]

        for row in table_data[1:]:
            date, title, authors, code_url, summary_zh, summary_en = (
                str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5])
            )
            summary = summary_zh if summary_zh and summary_zh != summary_en else (summary_en[:200] + ("…" if len(summary_en) > 200 else ""))
            code_part = f"  💻 [code]({code_url})" if code_url else ""
            entry = f"• [{title}]({row[6] if len(row) > 6 else ''})  ✍️ {authors}{code_part}\n  📝 {summary}"
            elements.append({"tag": "div", "text": {"tag": "lark_md", "content": entry}})

        data = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {"title": {"content": "📄 eess.IV Daily"}, "template": "blue"},
                "elements": elements,
            },
        }
        r = requests.post(webhook_url, json=data, timeout=10)
        logging.info(f"Feishu status: {r.status_code}")
    except Exception as e:
        logging.warning(f"Failed to send Feishu message: {e}")


def generate_feishu_table(data_collector: list, date_str: str) -> list:
    """
    返回飞书表格数据（行结构）:
    [date, title, authors, code_url, summary_zh, summary_en, arxiv_url]
    """
    table_data = [["Date", "Title", "Authors", "Code", "摘要(zh)", "摘要(en)", "URL"]]

    all_papers = []
    for data in data_collector:
        for topic, papers in data.items():
            for paper_id, paper_info in papers.items():
                if not isinstance(paper_info, dict):
                    continue
                all_papers.append({
                    'date': paper_info.get('update_time', ''),
                    'title': paper_info.get('title', ''),
                    'authors': paper_info.get('first_author', ''),
                    'code_url': paper_info.get('code_url', ''),
                    'summary_zh': paper_info.get('summary_zh', ''),
                    'summary_en': paper_info.get('summary_en', ''),
                    'url': paper_info.get('url', ''),
                })

    all_papers.sort(key=lambda x: x['date'], reverse=True)
    for p in all_papers:
        table_data.append([
            p['date'], p['title'], f"{p['authors']} et.al.",
            p['code_url'], p['summary_zh'], p['summary_en'], p['url'],
        ])
    return table_data


# ----------------------------------------------------------------------
# JSON 持久化
# ----------------------------------------------------------------------

def update_json_file(filename, data_dict):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(filename, "r") as f:
        content = f.read()
        m = json.loads(content) if content else {}

    json_data = m.copy()
    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]
            if keyword in json_data.keys():
                json_data[keyword].update(papers)
            else:
                json_data[keyword] = papers

    with open(filename, "w") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------
# README 渲染
# ----------------------------------------------------------------------

def _escape_md(s: str) -> str:
    """转义 Markdown 表格元字符 + HTML 尖括号"""
    if s is None:
        return ""
    return (s.replace("|", "\\|")
             .replace("\n", " ")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def _format_summary_cell(p: dict, max_inline: int = 120, en_fallback_chars: int = 240) -> str:
    summary_zh = p.get("summary_zh") or ""
    summary_en = p.get("summary_en") or ""

    if summary_zh and summary_zh != summary_en:
        # 有中文翻译
        if len(summary_zh) <= max_inline:
            return _escape_md(summary_zh)
        return (f"<details><summary>展开中文摘要 ({len(summary_zh)} 字)</summary>"
                f"{_escape_md(summary_zh)}</details>")

    # 翻译失败：回退到英文摘要
    truncated = summary_en if len(summary_en) <= en_fallback_chars else summary_en[:en_fallback_chars].rsplit(" ", 1)[0] + "…"
    return _escape_md(truncated)


def json_to_md(filename, md_filename, show_badge: bool = True):
    with open(filename, "r") as f:
        content = f.read()
        data = json.loads(content) if content else {}

    DateNow = datetime.date.today().strftime("%Y.%m.%d")

    all_papers = []
    for topic, day_content in data.items():
        if not day_content:
            continue
        for _, v in day_content.items():
            if v is not None and isinstance(v, dict):
                all_papers.append((topic, v))
    all_papers.sort(key=lambda x: x[1].get('update_time', ''), reverse=True)

    os.makedirs(os.path.dirname(md_filename) or ".", exist_ok=True)
    with open(md_filename, "w+") as f:
        pass  # truncate

    with open(md_filename, "a+") as f:
        f.write(f"## Updated on {DateNow}\n\n")
        f.write(f"> 来源：arXiv **eess.IV** (Image and Video Processing) · 每日 09:00 北京时间自动更新\n")
        f.write(f"> 邮件订阅：见 [使用说明](./docs/README.md#邮件订阅)\n\n")
        f.write(f"|Publish Date|Title|Authors|Code|中文摘要|\n")
        f.write(f"|---|---|---|---|---|\n")

        for _, v in all_papers:
            paper_url = v.get('url', '')
            arxiv_id = paper_url.split('/')[-1] if paper_url else ''
            title_md = f"[{_escape_md(v.get('title', ''))}]({paper_url})" if paper_url else _escape_md(v.get('title', ''))
            code_url = v.get('code_url', '')
            code_md = f"[GitHub]({code_url})" if code_url else "—"
            summary_cell = _format_summary_cell(v)

            row = (f"|**{v.get('update_time', '')}**|{title_md}|"
                   f"{_escape_md(v.get('first_author', ''))} et.al.|{code_md}|{summary_cell}|\n")
            f.write(row)

        anchor = f"updated-on-{DateNow.replace('.', '').lower()}"
        f.write(f"\n<p align=right>(<a href=#{anchor}>back to top</a>)</p>\n\n")

        if show_badge:
            f.write("[contributors-shield]: https://img.shields.io/github/contributors/bitchenhui/compress-arxiv-daily.svg?style=for-the-badge\n")
            f.write("[contributors-url]: https://github.com/bitchenhui/compress-arxiv-daily/graphs/contributors\n")
            f.write("[forks-shield]: https://img.shields.io/github/forks/bitchenhui/compress-arxiv-daily.svg?style=for-the-badge\n")
            f.write("[forks-url]: https://github.com/bitchenhui/compress-arxiv-daily/network/members\n")
            f.write("[stars-shield]: https://img.shields.io/github/stars/bitchenhui/compress-arxiv-daily.svg?style=for-the-badge\n")
            f.write("[stars-url]: https://github.com/bitchenhui/compress-arxiv-daily/stargazers\n")
            f.write("[issues-shield]: https://img.shields.io/github/issues/bitchenhui/compress-arxiv-daily.svg?style=for-the-badge\n")
            f.write("[issues-url]: https://github.com/bitchenhui/compress-arxiv-daily/issues\n\n")

    logging.info(f"README rendered: {md_filename} ({len(all_papers)} papers)")


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

def _resolve_date(config) -> tuple:
    """
    返回 (date_from, date_to)。

    优先级：
      1. CLI --date 参数（本地手动测试用）
      2. 默认：Asia/Shanghai 当天日期
    """
    if config.get('search_date'):
        sd = config['search_date']
        try:
            datetime.datetime.strptime(sd, '%Y%m%d')
            return sd, sd
        except ValueError:
            logging.error(f"Invalid search_date: {sd}, fallback to today (Asia/Shanghai)")

    # 默认：当天（Asia/Shanghai 视角，UTC+8）
    beijing_now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    today = beijing_now.date()
    return today.strftime('%Y%m%d'), today.strftime('%Y%m%d')


def demo(**config):
    global _translate_count_today
    _translate_count_today = 0  # 每次执行重置

    is_history = config.get('update_history', False)
    feishu_webhook = config.get('feishu_webhook', '')

    if is_history:
        category = config.get('arxiv_category', 'eess.IV')
        min_citations = config.get('min_citations', 0)
        history_date_from = config.get('history_date_from', '').strip()
        history_date_to = config.get('history_date_to', '').strip()

        if not history_date_from:
            logging.error("history_date_from is required for history search")
            return

        date_to = history_date_to if history_date_to else None
        date_range = f"{history_date_from} to {history_date_to or 'present'}"

        logging.info(f"GET history papers ({date_range}, min_citations={min_citations}) begin")
        history_file = config.get('json_history_path', './docs/eess-iv-history.json')
        history_collector = []
        # 历史模式：直接按类目 cat:eess.IV 拉
        data = get_history_papers(
            category=category,
            query=f"cat:{category}",
            max_results=100,
            date_from=history_date_from,
            date_to=date_to,
            min_citations=min_citations,
        )
        history_collector.append(data)
        update_history_json(history_file, history_collector)
        logging.info(f"GET history papers end")
        return

    # 每日模式
    category = config.get('arxiv_category', 'eess.IV')
    date_from, date_to = _resolve_date(config)
    logging.info(f"Searching {category} papers for {date_from}")

    data, data_web = get_daily_papers(
        category=category,
        max_results=config.get('max_results', 300),
        date_from=date_from,
        date_to=date_to,
        exclude_keywords=config.get('medical_exclude_keywords', []),
        github_cfg=config.get('github', {}),
        translation_cfg=config.get('translation', {}),
    )

    if config.get('publish_readme', True):
        json_file = config['json_readme_path']
        md_file = config['md_readme_path']
        update_json_file(json_file, [data])
        json_to_md(json_file, md_file, show_badge=config.get('show_badge', True))

    # 飞书推送
    if feishu_webhook:
        date_now = datetime.date.today().strftime("%Y-%m-%d")
        table_data = generate_feishu_table([data], date_now)
        if len(table_data) > 1:
            send_to_feishu(feishu_webhook, table_data)
        else:
            send_no_papers_message(feishu_webhook)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml',
                        help='configuration file path')
    parser.add_argument('--update_history', default=False,
                        action="store_true",
                        help='whether to update history papers with high citations')
    parser.add_argument('--date', type=str, default=None,
                        help='specify search date in YYYYMMDD format (e.g., 20260305)')
    args = parser.parse_args()

    config = load_config(args.config_path)
    config = {**config, 'update_history': args.update_history, 'search_date': args.date}
    demo(**config)
