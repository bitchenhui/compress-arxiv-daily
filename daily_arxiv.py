import os
import re
import json
import time
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

def load_config(config_file:str) -> dict:
    '''
    config_file: input config file path
    return: a dict of configuration
    '''
    # make filters pretty
    def pretty_filters(**config) -> dict:
        keywords = dict()
        EXCAPE = '\"'
        QUOTA = '' # NO-USE
        OR = ' OR ' # TODO
        def parse_filters(filters:list):
            ret = ''
            for idx in range(0,len(filters)):
                filter = filters[idx]
                if len(filter.split()) > 1:
                    ret += (EXCAPE + filter + EXCAPE)
                else:
                    ret += (QUOTA + filter + QUOTA)
                if idx != len(filters) - 1:
                    ret += OR
            return ret
        for k,v in config['keywords'].items():
            keywords[k] = parse_filters(v['filters'])
        return keywords
    with open(config_file,'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        config['kv'] = pretty_filters(**config)
        logging.info(f'config = {config}')
    return config

def get_authors(authors, first_author = False):
    output = str()
    if first_author == False:
        output = ", ".join(str(author) for author in authors)
    else:
        output = str(authors[0])
    return output
def sort_papers(papers):
    output = dict()
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output

def check_relevance(title: str, abstract: str, filters: list, topic: str) -> bool:
    """
    Check if paper is relevant based on keywords in title and abstract
    @param title: paper title
    @param abstract: paper abstract
    @param filters: list of keywords to match
    @param topic: the topic/category name for topic-specific logic
    @return: True if relevant
    """
    text = (title + " " + abstract).lower()
    title_lower = title.lower()
    
    # For Video Codec, require stricter matching
    if topic == "Video Codec":
        # Must have video/coding related terms in title or abstract
        video_terms = ["video", "codec", "coding", "compression", "encoder", "decoder", "bitrate", "hevc", "h.264", "h.265", "av1", "vvc", "mpeg"]
        has_video = any(term in text for term in video_terms)
        has_coding = any(term in text for term in ["codec", "coding", "compression", "encoder", "decoder"])
        # Must match at least 2 of the specific codec keywords
        codec_keywords = [k.lower() for k in filters if len(k) > 2]
        keyword_matches = sum(1 for k in codec_keywords if k in text)
        return has_video and (has_coding or keyword_matches >= 2)
    
    # For other topics, require at least 2 keyword matches or title match
    match_count = 0
    for keyword in filters:
        keyword_lower = keyword.lower()
        # For multi-word keywords, check if any significant word appears
        if ' ' in keyword_lower:
            parts = keyword_lower.split()
            for part in parts:
                if len(part) > 3 and part in text:
                    match_count += 1
        else:
            if len(keyword_lower) > 2 and keyword_lower in text:
                match_count += 1
    
    # Check title for any keyword
    title_match = any(k.lower() in title_lower for k in filters)
    return match_count >= 2 or title_match

def get_paper_citations(arxiv_id: str) -> int:
    """
    Get citation count from Semantic Scholar API
    @param arxiv_id: str, e.g., "2108.09112"
    @return: citation count
    """
    try:
        # Remove version number if present
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        url = f"{semantic_scholar_url}{arxiv_id}?fields=citationCount"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('citationCount', 0)
        else:
            return 0
    except Exception as e:
        logging.warning(f"Failed to get citations for {arxiv_id}: {e}")
        return 0

def search_arxiv_with_retry(query: str, max_results: int = 10, max_retries: int = 3, date_from: str = None, date_to: str = None):
    """
    Search arXiv with retry logic for rate limiting
    @param date_from: search from date (YYYYMMDD)
    @param date_to: search to date (YYYYMMDD)
    """
    # Build date filter if provided
    date_filter = ""
    if date_from and date_to:
        date_filter = f"submittedDate:[{date_from} TO {date_to}]"
        query = f"({query}) AND {date_filter}" if query else date_filter
    
    for attempt in range(max_retries):
        try:
            search_engine = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            # Try to get results to trigger any early errors
            results = list(search_engine.results())
            return results
        except Exception as e:
            error_msg = str(e)
            if "Rate exceeded" in error_msg or "429" in error_msg:
                wait_time = (attempt + 1) * 30  # 30, 60, 90 seconds
                logging.warning(f"Rate limit hit, waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logging.error(f"Arxiv search error: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(10)
    return []

def get_history_papers(topic, query: str, min_citations: int = 500, max_results: int = 50):
    """
    Get historical papers with high citations
    """
    content = dict()
    for attempt in range(3):
        try:
            search_engine = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            break
        except Exception as e:
            if attempt < 2:
                logging.warning(f"Rate limit hit, waiting 60s before retry...")
                time.sleep(60)
            else:
                logging.error(f"Failed to search arxiv: {e}")
                return {topic: {}}

    for result in search_engine.results():
        paper_id = result.get_short_id()
        # Remove version number
        ver_pos = paper_id.find('v')
        if ver_pos != -1:
            paper_key = paper_id[0:ver_pos]
        else:
            paper_key = paper_id

        # Get citation count
        citation_count = get_paper_citations(paper_key)
        logging.info(f"Paper {paper_key}: {result.title[:50]}... citations={citation_count}")

        if citation_count >= min_citations:
            paper_url = arxiv_url + 'abs/' + paper_key
            paper_first_author = get_authors(result.authors, first_author=True)

            content[paper_key] = {
                "title": result.title,
                "authors": get_authors(result.authors),
                "first_author": paper_first_author,
                "url": paper_url,
                "citations": citation_count,
                "published": result.published.date().isoformat()
            }
            logging.info(f"  -> Added to history (citations: {citation_count})")

        # Rate limit between citation API calls
        time.sleep(3)

    return {topic: content}

def update_history_json(filename: str, data_dict: dict):
    """
    Update history JSON file with new papers (merge, avoid duplicates)
    """
    with open(filename, "r") as f:
        content = f.read()
        if not content:
            existing_data = {}
        else:
            existing_data = json.loads(content)

    # Merge new data
    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]
            if keyword not in existing_data:
                existing_data[keyword] = {}
            # Merge: keep the one with higher citations if duplicate
            for paper_id, paper_info in papers.items():
                if paper_id in existing_data[keyword]:
                    existing_citations = existing_data[keyword][paper_id].get('citations', 0)
                    if paper_info['citations'] > existing_citations:
                        existing_data[keyword][paper_id] = paper_info
                else:
                    existing_data[keyword][paper_id] = paper_info

    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    logging.info(f"History updated: {filename}")

def send_to_feishun(webhook_url: str, content: str):
    """
    Send message to Feishu webhook
    """
    try:
        headers = {"Content-Type": "application/json"}
        data = {"msg_type": "text", "content": {"text": content}}
        response = requests.post(webhook_url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            logging.info("Feishu message sent successfully")
        else:
            logging.warning(f"Failed to send Feishu message: {response.status_code}")
    except Exception as e:
        logging.warning(f"Failed to send Feishu message: {e}")

def generate_feishu_message(data_collector: list, date_str: str) -> str:
    """
    Generate Feishu message from daily papers (标题, 作者, 分类, 链接)
    """
    message = f"📄 Compress ArXiv Daily - {date_str}\n\n"

    for data in data_collector:
        for topic, papers in data.items():
            if not papers:
                continue
            message += f"### {topic}\n"
            # Sort by date (newest first)
            # Handle both dict and string formats
            try:
                sorted_papers = sorted(papers.items(), key=lambda x: x[1].get('update_time', '') if isinstance(x[1], dict) else '', reverse=True)
            except:
                sorted_papers = list(papers.items())[:10]

            for paper_id, paper_info in sorted_papers[:10]:  # Top 10 per category
                # Handle dict format
                if isinstance(paper_info, dict):
                    title = paper_info.get('title', '')[:60]
                    authors = paper_info.get('first_author', '')
                    category = paper_info.get('category', 'cs.CV')
                    url = paper_info.get('url', '')
                else:
                    # Fallback for string format
                    title = str(paper_info)[:60]
                    authors = ''
                    category = 'cs.CV'
                    url = ''
                message += f"• {title}\n"
                message += f"  作者: {authors} et.al. | 分类: {category}\n"
                message += f"  链接: {url}\n\n"
            message += "\n"

    return message

def get_daily_papers(topic, query: str, max_results: int = 10, date_from: str = None, date_to: str = None, filters: list = None):
    """
    @param topic: str
    @param query: str
    @param date_from: search from date (YYYYMMDD)
    @param date_to: search to date (YYYYMMDD)
    @param filters: list of keywords for relevance filtering
    @return paper_with_code: dict
    """
    # Wait before search to avoid rate limit
    time.sleep(3)

    content = dict()
    content_to_web = dict()

    # Use retry logic with date filter
    results = search_arxiv_with_retry(query, max_results=max_results, date_from=date_from, date_to=date_to)

    for result in results:
        # Check relevance if filters provided
        if filters:
            abstract = result.summary.replace("\n", " ")
            if not check_relevance(result.title, abstract, filters, topic):
                logging.info(f"Skipping irrelevant paper: {result.title[:50]}...")
                continue
        paper_id = result.get_short_id()
        paper_title = result.title
        paper_url = result.entry_id
        paper_abstract = result.summary.replace("\n", " ")
        paper_authors = get_authors(result.authors)
        paper_first_author = get_authors(result.authors, first_author=True)
        primary_category = result.primary_category
        publish_time = result.published.date()
        update_time = result.updated.date()
        comments = result.comment

        logging.info(f"Time = {update_time} title = {paper_title} author = {paper_first_author}")

        # eg: 2108.09112v1 -> 2108.09112
        ver_pos = paper_id.find('v')
        if ver_pos == -1:
            paper_key = paper_id
        else:
            paper_key = paper_id[0:ver_pos]
        paper_url = arxiv_url + 'abs/' + paper_key

        content[paper_key] = {
            "update_time": str(update_time),
            "title": paper_title,
            "first_author": paper_first_author,
            "authors": paper_authors,
            "category": primary_category,
            "url": paper_url,
            "code_url": ""
        }

        content_to_web[paper_key] = "- {}, **{}**, {} et.al., [{}]({})".format(
               update_time, paper_title, paper_first_author, primary_category, paper_url)

        comments = None
        if comments is not None:
            content_to_web[paper_key] += f", {comments}\n"
        else:
            content_to_web[paper_key] += f"\n"

    data = {topic: content}
    data_web = {topic: content_to_web}
    return data, data_web

def update_paper_links(filename):
    '''
    weekly update paper links in json file
    '''
    def parse_arxiv_string(s):
        parts = s.split("|")
        date = parts[1].strip()
        title = parts[2].strip()
        authors = parts[3].strip()
        arxiv_id = parts[4].strip()
        code = parts[5].strip()
        arxiv_id = re.sub(r'v\d+', '', arxiv_id)
        return date, title, authors, arxiv_id, code

    with open(filename, "r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

        json_data = m.copy()

        for keywords, v in json_data.items():
            logging.info(f'keywords = {keywords}')
            for paper_id, contents in v.items():
                contents = str(contents)

                update_time, paper_title, paper_first_author, paper_url, code_url = parse_arxiv_string(contents)

                contents = "|{}|{}|{}|{}|{}|\n".format(update_time, paper_title, paper_first_author, paper_url, code_url)
                json_data[keywords][paper_id] = str(contents)
                logging.info(f'paper_id = {paper_id}, contents = {contents}')

                logging.info(f'Skipping code link update for paper_id = {paper_id} (PapersWithCode API deprecated)')

        with open(filename, "w") as f:
            json.dump(json_data, f, ensure_ascii=False)

def update_json_file(filename, data_dict):
    '''
    daily update json file using data_dict
    '''
    with open(filename, "r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

    json_data = m.copy()

    # update papers in each keywords
    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]

            if keyword in json_data.keys():
                json_data[keyword].update(papers)
            else:
                json_data[keyword] = papers

    with open(filename, "w") as f:
        json.dump(json_data, f, ensure_ascii=False)

def json_to_md(filename, md_filename,
               task='',
               to_web=False,
               use_title=True,
               use_tc=True,
               show_badge=True,
               use_b2t=True):
    """
    @param filename: str
    @param md_filename: str
    @return None
    """
    def pretty_math(s: str) -> str:
        ret = ''
        match = re.search(r"\$.*\$", s)
        if match is None:
            return s
        math_start, math_end = match.span()
        space_trail = space_leading = ''
        if s[:math_start][-1] != ' ' and '*' != s[:math_start][-1]:
            space_trail = ' '
        if s[math_end:][0] != ' ' and '*' != s[math_end:][0]:
            space_leading = ' '
        ret += s[:math_start]
        ret += f'{space_trail}${match.group()[1:-1].strip()}${space_leading}'
        ret += s[math_end:]
        return ret

    DateNow = datetime.date.today()
    DateNow = str(DateNow)
    DateNow = DateNow.replace('-', '.')

    with open(filename, "r") as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)

    # clean README.md if daily already exist else create it
    with open(md_filename, "w+") as f:
        pass

    # write data into README.md
    with open(md_filename, "a+") as f:

        if (use_title == True) and (to_web == True):
            f.write("---\n" + "layout: default\n" + "---\n\n")

        if use_title == True:
            f.write("## Updated on " + DateNow + "\n")
        else:
            f.write("> Updated on " + DateNow + "\n")

        f.write("> Usage instructions: [here](./docs/README.md#usage)\n\n")

        # Add: table of contents
        if use_tc == True:
            f.write("<details>\n")
            f.write("  <summary>Table of Contents</summary>\n")
            f.write("  <ol>\n")
            for keyword in data.keys():
                day_content = data[keyword]
                if not day_content:
                    continue
                kw = keyword.replace(' ', '-')
                f.write(f"    <li><a href=#{kw.lower()}>{keyword}</a></li>\n")
            f.write("  </ol>\n")
            f.write("</details>\n\n")

        # Collect all papers into one list
        all_papers = []
        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue
            # sort papers by date
            day_content = sort_papers(day_content)
            for _, v in day_content.items():
                if v is not None and isinstance(v, dict):
                    all_papers.append((keyword, v))
        
        # Sort all papers by date (newest first)
        all_papers.sort(key=lambda x: x[1].get('update_time', ''), reverse=True)
        
        # Write combined table header (no ## sections)
        if use_title == True:
            if to_web == False:
                f.write("|Publish Date|Title|Authors|Tag|PDF|\n" + "|---|---|---|---|---|\n")
            else:
                f.write("| Publish Date | Title | Authors | Tag | PDF |\n")
                f.write("|:---------|:-----------------------|:---------|:------|:------|\n")
        
        # Write all papers in one table
        for keyword, v in all_papers:
            paper_url = v.get('url', '')
            arxiv_id = paper_url.split('/')[-1] if paper_url else ''
            pdf_url = paper_url.replace('abs', 'pdf')
            pdf_col = f"[{arxiv_id}]({pdf_url})"
            row = "|**{}**|**{}**|{} et.al.|{}|{}|\n".format(
                v.get('update_time', ''),
                v.get('title', ''),
                v.get('first_author', ''),
                keyword,
                pdf_col
            )
            f.write(pretty_math(row))

        # Add: back to top
        if use_b2t:
            top_info = f"#Updated on {DateNow}"
            top_info = top_info.replace(' ', '-').replace('.', '')
            f.write(f"<p align=right>(<a href={top_info.lower()}>back to top</a>)</p>\n\n")

        if show_badge == True:
            # we don't like long string, break it!
            f.write((f"[contributors-shield]: https://img.shields.io/github/"
                     f"contributors/bitchenhui/compress-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[contributors-url]: https://github.com/bitchenhui/"
                     f"compress-arxiv-daily/graphs/contributors\n"))
            f.write((f"[forks-shield]: https://img.shields.io/github/forks/bitchenhui/"
                     f"compress-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[forks-url]: https://github.com/bitchenhui/"
                     f"compress-arxiv-daily/network/members\n"))
            f.write((f"[stars-shield]: https://img.shields.io/github/stars/bitchenhui/"
                     f"compress-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[stars-url]: https://github.com/bitchenhui/"
                     f"compress-arxiv-daily/stargazers\n"))
            f.write((f"[issues-shield]: https://img.shields.io/github/issues/bitchenhui/"
                     f"compress-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[issues-url]: https://github.com/bitchenhui/"
                     f"compress-arxiv-daily/issues\n\n"))

    logging.info(f"{task} finished")

def demo(**config):
    data_collector = []
    data_collector_web = []

    keywords = config['kv']
    max_results = config.get('max_results', 100)
    publish_readme = config['publish_readme']
    publish_gitpage = config['publish_gitpage']
    publish_wechat = config['publish_wechat']
    show_badge = config['show_badge']
    
    # Calculate yesterday's date for filtering (YYYYMMDD format)
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    date_from = yesterday.strftime('%Y%m%d')
    date_to = yesterday.strftime('%Y%m%d')
    logging.info(f"Searching papers from {date_from} to {date_to}")
    
    logging.info(f"GET daily papers begin")
    for topic, keyword in keywords.items():
        # Get filters from config for relevance check
        topic_filters = config.get('keywords', {}).get(topic, {}).get('filters', [])
        logging.info(f"Keyword: {topic}")
        data, data_web = get_daily_papers(topic, query=keyword,
                                          max_results=max_results,
                                          date_from=date_from,
                                          date_to=date_to,
                                          filters=topic_filters)
        data_collector.append(data)
        data_collector_web.append(data_web)
        print("\n")
        # Wait between each keyword to avoid rate limit
        time.sleep(5)
    logging.info(f"GET daily papers end")

    # 1. update README.md file
    if publish_readme:
        json_file = config['json_readme_path']
        md_file = config['md_readme_path']
        update_json_file(json_file, data_collector)
        json_to_md(json_file, md_file, task='Update Readme', \
                    show_badge=show_badge)

    # 2. update docs/index.md file (to gitpage)
    if publish_gitpage:
        json_file = config['json_gitpage_path']
        md_file = config['md_gitpage_path']
        update_json_file(json_file, data_collector)
        json_to_md(json_file, md_file, task='Update GitPage', \
                    to_web=True, show_badge=show_badge, \
                    use_tc=False, use_b2t=False)

    # 3. Update docs/wechat.md file
    if publish_wechat:
        json_file = config['json_wechat_path']
        md_file = config['md_wechat_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector_web)
        json_to_md(json_file, md_file, task='Update Wechat', \
                    to_web=False, use_title=False, show_badge=show_badge)

    # 4. Update history papers (high citation > 500)
    if config.get('update_history', False):
        logging.info(f"GET history papers (citations > 500) begin")
        history_file = config.get('json_history_path', './docs/compress-arxiv-history.json')
        history_collector = []
        for topic, keyword in keywords.items():
            logging.info(f"History Keyword: {topic}")
            history_data = get_history_papers(topic, query=keyword, min_citations=500, max_results=30)
            history_collector.append(history_data)
            # Wait between keywords
            time.sleep(10)
        update_history_json(history_file, history_collector)
        logging.info(f"GET history papers end")

    # 5. Send to Feishu
    feishu_webhook = config.get('feishu_webhook', '')
    if feishu_webhook and data_collector:
        date_now = datetime.date.today().strftime('%Y-%m-%d')
        message = generate_feishu_message(data_collector, date_now)
        send_to_feishun(feishu_webhook, message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml',
                        help='configuration file path')
    parser.add_argument('--update_history', default=False,
                        action="store_true", help='whether to update history papers with high citations')
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = {**config, 'update_history': args.update_history}
    demo(**config)
