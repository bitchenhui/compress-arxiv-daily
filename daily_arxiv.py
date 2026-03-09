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
    def pretty_filters(**config) -> dict:
        keywords = dict()
        EXCAPE = '\"'
        QUOTA = ''
        OR = ' OR '
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
    text = (title + " " + abstract).lower()
    title_lower = title.lower()

    # Video Codec: 传统视频/图像编解码算法优化
    # 必须包含：视频编解码核心标准/技术 + 编解码相关词
    if topic == "Video Codec":
        # 核心技术词
        core_terms = ["video", "hevc", "h.265", "h.264", "av1", "vvc", "h.266",
                      "inter prediction", "intra prediction", "motion estimation",
                      "cabac", "rate control", "rd optimization", "transform"]
        has_core = any(term in text for term in core_terms)

        # 编解码相关词
        codec_terms = ["codec", "coding", "compression", "encoder", "decoder", "bitrate"]
        has_codec = any(term in text for term in codec_terms)

        # 排除：视频理解、目标检测、分类等非编解码任务
        exclude_terms = ["object detection", "tracking", "recognition", "classification",
                        "segmentation", "generation", "synthesis", "super resolution"]
        has_exclude = any(term in text for term in exclude_terms)

        return has_core and has_codec and not has_exclude

    # Neural Compression: 机器学习视频/图像编解码（端到端学习压缩、DCVC、VAE、GAN等）
    # 必须包含：神经网络相关词 + 压缩/重建相关词
    if topic == "Neural Compression":
        neural_terms = ["neural", "learned", "deep learning", "cnn", "autoencoder",
                       "vae", "gan", "generative", "dcvc", "end-to-end", "latent",
                       "variational", "neural network", "convolution"]
        has_neural = any(term in text for term in neural_terms)

        compression_terms = ["compression", "codec", "coding", "encoder", "decoder",
                           "reconstruction", "rate", "bits", "entropy", "image compression", "video compression"]
        has_compression = any(term in text for term in compression_terms)

        # 排除：视频理解、分类等
        exclude_terms = ["object detection", "tracking", "recognition", "action recognition",
                        "video understanding", "video caption", "video classification"]
        has_exclude = any(term in text for term in exclude_terms)

        return has_neural and has_compression and not has_exclude

    # MLLM: LLM/多模态与视频/图像编解码交叉
    # 必须包含：LLM/多模态相关词 + 视频/压缩/通信相关词
    if topic == "MLLM":
        llm_terms = ["llm", "large language model", "multimodal", "vlm", "vision language",
                    "gpt", "transformer", "semantic", "token", "diffusion model",
                    "language model", "chatgpt", "bert", "vision transformer"]
        has_llm = any(term in text for term in llm_terms)

        video_terms = ["video", "image", "compression", "codec", "coding", "communication",
                      "bit", "rate", "semantic communication", "tokenize"]
        has_video = any(term in text for term in video_terms)

        return has_llm and has_video

    # Video Quality: 视频/图像质量评估
    if topic == "Video Quality":
        quality_terms = ["quality", "ssim", "psnr", "vmaf", "metric", "assessment",
                        "perceptual", "no-reference", "full-reference", "blind",
                        "quality assessment", "quality evaluation"]
        has_quality = any(term in text for term in quality_terms)

        media_terms = ["video", "image", "frame", "picture", "picture quality", "video quality"]
        has_media = any(term in text for term in media_terms)

        return has_quality and has_media

    # 3D Compression: 新兴场景技术（点云、VR、360、光场、NeRF等）
    if topic == "3D Compression":
        # 新兴技术词
        emerging_terms = ["point cloud", "voxel", "mesh", "gaussian", "splatting",
                         "vr", "virtual reality", "360", "omnidirectional", "panoramic",
                         "light field", "nerf", "radiance field", "multiview",
                         "stereo", "depth map", "mvd", "volumetric", "immersive",
                         "holographic", "plenoptic"]
        has_emerging = any(term in text for term in emerging_terms)

        # 压缩相关词
        compression_terms = ["compression", "codec", "coding", "encoding", "coding"]
        has_compression = any(term in text for term in compression_terms)

        return has_emerging and has_compression

    # Default fallback
    match_count = sum(1 for k in filters if k.lower() in text)
    title_match = any(k.lower() in title_lower for k in filters)
    return match_count >= 1 or title_match

def get_paper_citations(arxiv_id: str, retry: int = 3) -> int:
    arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
    url = f"{semantic_scholar_url}{arxiv_id}?fields=citationCount"

    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('citationCount', 0)
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
    """
    Batch fetch citations for multiple papers using Semantic Scholar graph API
    Returns dict mapping arxiv_id to citation count
    """
    if not arxiv_ids:
        return {}

    # Clean up arxiv_ids - add arxiv: prefix as required by API
    clean_ids = [re.sub(r'v\d+$', '', aid) for aid in arxiv_ids]
    # Add arxiv: prefix for each ID
    prefixed_ids = [f"ARXIV:{aid.upper()}" for aid in clean_ids]

    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params = {
        "fields": "paperId,citationCount,externalIds"
    }
    # Send as JSON with "ids" field
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
                # Rate limited, wait and retry
                wait_time = (attempt + 1) * 30
                logging.warning(f"Batch rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Log error details for debugging
                logging.warning(f"Batch request failed: {response.status_code}, response: {response.text[:200]}")
                time.sleep(5)
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(5)
            else:
                logging.warning(f"Batch citations failed: {e}")
                return {}

    return {}

def search_arxiv_with_retry(query: str, max_results: int = 10, max_retries: int = 3, date_from: str = None, date_to: str = None):
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
            results = list(search_engine.results())
            return results
        except Exception as e:
            error_msg = str(e)
            if "Rate exceeded" in error_msg or "429" in error_msg:
                wait_time = (attempt + 1) * 30
                logging.warning(f"Rate limit hit, waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logging.error(f"Arxiv search error: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(10)
    return []

def get_history_papers(topic, query: str, max_results: int = 100, date_from: str = "20100101", date_to: str = None, min_citations: int = 0):
    content = dict()
    papers_with_citations = []

    if not date_to:
        date_to = datetime.date.today().strftime('%Y%m%d')
    date_filter = f"submittedDate:[{date_from} TO {date_to}]"
    full_query = f"({query}) AND {date_filter}"

    # If min_citations is set, search more to find enough papers with high citations
    if min_citations > 0:
        # Search through all results to find papers with high citations
        # Limit to 3000 to avoid taking too long
        search_max = 1000
    else:
        search_max = max_results

    logging.info(f"History search: {date_from} to {date_to}, min_citations={min_citations}, searching up to {search_max} papers to find all with citations > {min_citations}")

    for attempt in range(3):
        try:
            search_engine = arxiv.Search(
                query=full_query,
                max_results=search_max,
                sort_by=arxiv.SortCriterion.Relevance  # Sort by relevance
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
                return {topic: {}}

    # Collect paper keys first, also get filters for relevance check
    topic_filters = []
    # Get filters from config based on topic
    from yaml import safe_load
    with open('config.yaml', 'r') as f:
        config_data = safe_load(f)
        topic_filters = config_data.get('keywords', {}).get(topic, {}).get('filters', [])

    paper_results = []
    irrelevant_count = 0
    for result in search_engine.results():
        paper_id = result.get_short_id()
        ver_pos = paper_id.find('v')
        paper_key = paper_id[0:ver_pos] if ver_pos != -1 else paper_id

        # Check relevance BEFORE fetching citations (save time)
        if topic_filters:
            abstract = result.summary.replace("\n", " ")
            if not check_relevance(result.title, abstract, topic_filters, topic):
                irrelevant_count += 1
                continue

        paper_results.append((result, paper_key))

    logging.info(f"Filtered out {irrelevant_count} irrelevant papers, keeping {len(paper_results)} relevant papers")

    # Fetch all citations in batch (max 100 per request)
    paper_keys = [pk for _, pk in paper_results]
    logging.info(f"Fetching citations for {len(paper_keys)} papers in batch...")

    citations_dict = {}
    batch_size = 100
    for i in range(0, len(paper_keys), batch_size):
        batch = paper_keys[i:i+batch_size]
        logging.info(f"Fetching batch {i//batch_size + 1}/{(len(paper_keys)-1)//batch_size + 1}...")
        batch_result = get_papers_citations_batch(batch)
        citations_dict.update(batch_result)
        time.sleep(1)  # Rate limiting between batches

    # Build papers with citations
    papers_with_citations = []
    for result, paper_key in paper_results:
        citation_count = citations_dict.get(paper_key, 0)
        papers_with_citations.append({
            'result': result,
            'paper_key': paper_key,
            'citations': citation_count
        })

    # Sort by citations descending (high to low)
    papers_with_citations.sort(key=lambda x: x['citations'], reverse=True)

    # Filter by minimum citations if specified
    if min_citations > 0:
        papers_with_citations = [p for p in papers_with_citations if p['citations'] >= min_citations]
        # Keep all papers that meet the criteria, not limited by max_results
        top_papers = papers_with_citations
    else:
        top_papers = papers_with_citations[:max_results]

    logging.info(f"Found {len(top_papers)} papers with citations >= {min_citations}")

    for item in top_papers:
        result = item['result']
        paper_key = item['paper_key']
        citation_count = item['citations']
        logging.info(f"Paper {paper_key}: {result.title[:50]}... citations={citation_count}")

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

    return {topic: content}

def update_history_json(filename: str, data_dict: dict):
    with open(filename, "r") as f:
        content = f.read()
        if not content:
            existing_data = {}
        else:
            existing_data = json.loads(content)

    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]
            if keyword not in existing_data:
                existing_data[keyword] = {}
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

def send_no_papers_message(webhook_url: str):
    """
    Send message when there are no new papers
    """
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": True
                },
                "header": {
                    "title": {
                        "content": "📄 Compress ArXiv Daily"
                    },
                    "template": "blue"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": "**今天没有增量论文！！！**"
                        }
                    }
                ]
            }
        }
        response = requests.post(webhook_url, headers=headers, json=data, timeout=10)
        logging.info(f"Feishu response status: {response.status_code}")
        if response.status_code == 200:
            logging.info("Feishu 'no papers' message sent successfully")
        else:
            logging.warning(f"Failed to send Feishu message: {response.status_code}")
    except Exception as e:
        logging.warning(f"Failed to send Feishu message: {e}")


def send_to_feishun(webhook_url: str, table_data: list):
    """
    Send message to Feishu webhook using interactive card with elegant formatting
    """
    try:
        headers = {"Content-Type": "application/json"}

        if not table_data or len(table_data) < 2:
            logging.warning("No table data to send")
            return

        # Get date from first paper for the header
        first_date = str(table_data[1][0]) if len(table_data) > 1 else ""
        paper_count = len(table_data) - 1

        # Build elements for the card
        elements = []

        # Add header info
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"📅 **{first_date}**  |  📚 **{paper_count}** 篇论文"
            }
        })

        # Process data rows - group by tag for better organization
        current_tag = None
        tag_index = {}  # Track current index within each tag
        for row in table_data[1:]:
            date = str(row[0])
            title = str(row[1])
            authors = str(row[2])
            tag = str(row[3])
            pdf_url = str(row[4])

            # Initialize tag index if needed
            if tag not in tag_index:
                tag_index[tag] = 0
            tag_index[tag] += 1

            # Add tag header if tag changes (compact - no extra spacing)
            if tag != current_tag:
                elements.append({
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**🏷️ {tag}**"
                    }
                })
                current_tag = tag

            # Full title (no truncation)
            display_title = title

            # Paper entry - compact format
            entry = f"**{tag_index[tag]}.** [{display_title}]({pdf_url})  ✍️ {authors}  📅 {date}"

            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": entry
                }
            })

        # Use interactive card with wide screen mode
        data = {
            "msg_type": "interactive",
            "card": {
                "config": {
                    "wide_screen_mode": True
                },
                "header": {
                    "title": {
                        "content": "📄 Compress ArXiv Daily"
                    },
                    "template": "blue"
                },
                "elements": elements
            }
        }

        response = requests.post(webhook_url, headers=headers, json=data, timeout=10)
        logging.info(f"Feishu response status: {response.status_code}")
        logging.info(f"Feishu response body: {response.text}")
        if response.status_code == 200:
            logging.info("Feishu message sent successfully")
        else:
            logging.warning(f"Failed to send Feishu message: {response.status_code}")
    except Exception as e:
        logging.warning(f"Failed to send Feishu message: {e}")

def generate_feishu_table(data_collector: list, date_str: str):
    """
    Generate table data for Feishu message
    Returns list of rows, each row is a list of cell values
    """
    # Header row
    table_data = [["Publish Date", "Title", "Authors", "Tag", "PDF"]]

    # Collect all papers
    all_papers = []
    for data in data_collector:
        for topic, papers in data.items():
            if not papers:
                continue
            try:
                sorted_papers = sorted(papers.items(), key=lambda x: x[1].get('update_time', '') if isinstance(x[1], dict) else '', reverse=True)
            except:
                sorted_papers = list(papers.items())

            for paper_id, paper_info in sorted_papers:
                if isinstance(paper_info, dict):
                    all_papers.append({
                        'date': paper_info.get('update_time', ''),
                        'title': paper_info.get('title', ''),
                        'authors': paper_info.get('first_author', ''),
                        'tag': topic,
                        'url': paper_info.get('url', '')
                    })

    all_papers.sort(key=lambda x: x['date'], reverse=True)

    # Add paper rows
    for paper in all_papers:
        date = paper['date']
        title = paper['title']
        authors = paper['authors']
        tag = paper['tag']
        url = paper['url']

        pdf_url = url.replace('abs', 'pdf') if url else ''

        # Full title (no truncation)
        table_data.append([date, title, f"{authors} et.al.", tag, pdf_url])

    return table_data

def get_daily_papers(topic, query: str, max_results: int = 10, date_from: str = None, date_to: str = None, filters: list = None):
    time.sleep(3)

    content = dict()
    content_to_web = dict()

    results = search_arxiv_with_retry(query, max_results=max_results, date_from=date_from, date_to=date_to)

    for result in results:
        if filters:
            abstract = result.summary.replace("\n", " ")
            if not check_relevance(result.title, abstract, filters, topic):
                logging.info(f"Skipping irrelevant paper: {result.title[:50]}...")
                continue
        paper_id = result.get_short_id()
        paper_title = result.title
        paper_url = result.entry_id
        paper_authors = get_authors(result.authors)
        paper_first_author = get_authors(result.authors, first_author=True)
        primary_category = result.primary_category
        update_time = result.updated.date()

        logging.info(f"Time = {update_time} title = {paper_title} author = {paper_first_author}")

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

        content_to_web[paper_key] = "- {}, **{}**, {} et.al., [{}]({})\n".format(
               update_time, paper_title, paper_first_author, primary_category, paper_url)

    data = {topic: content}
    data_web = {topic: content_to_web}
    return data, data_web

def update_json_file(filename, data_dict):
    with open(filename, "r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

    json_data = m.copy()

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

    with open(md_filename, "w+") as f:
        pass

    with open(md_filename, "a+") as f:

        if (use_title == True) and (to_web == True):
            f.write("---\n" + "layout: default\n" + "---\n\n")

        if use_title == True:
            f.write("## Updated on " + DateNow + "\n")
        else:
            f.write("> Updated on " + DateNow + "\n")

        f.write("> Usage instructions: [here](./docs/README.md#usage)\n\n")

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

        all_papers = []
        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue
            day_content = sort_papers(day_content)
            for _, v in day_content.items():
                if v is not None and isinstance(v, dict):
                    all_papers.append((keyword, v))

        all_papers.sort(key=lambda x: x[1].get('update_time', ''), reverse=True)

        if use_title == True:
            if to_web == False:
                f.write("|Publish Date|Title|Authors|Tag|PDF|\n" + "|---|---|---|---|---|\n")
            else:
                f.write("| Publish Date | Title | Authors | Tag | PDF |\n")
                f.write("|:---------|:-----------------------|:---------|:------|:------|\n")

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

        if use_b2t:
            top_info = f"#Updated on {DateNow}"
            top_info = top_info.replace(' ', '-').replace('.', '')
            f.write(f"<p align=right>(<a href={top_info.lower()}>back to top</a>)</p>\n\n")

        if show_badge == True:
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

    is_history_only = config.get('update_history', False)

    if is_history_only:
        min_citations = config.get('min_citations', 0)
        history_date_from = config.get('history_date_from', '').strip()
        history_date_to = config.get('history_date_to', '').strip()

        if not history_date_from:
            logging.error("history_date_from is required for history search")
            return

        date_from = history_date_from
        if history_date_to:
            date_to = history_date_to
            date_range = f"{date_from} to {date_to}"
        else:
            date_to = None  # Search to present
            date_range = f"{date_from}-present"

        logging.info(f"GET history papers ({date_range}, top 100 by citations, min_citations={min_citations}) begin")
        history_file = config.get('json_history_path', './docs/compress-arxiv-history.json')
        history_collector = []
        for topic, keyword_config in keywords.items():
            logging.info(f"History Keyword: {topic}")
            search_query = keyword_config.get('query', topic) if isinstance(keyword_config, dict) else topic
            history_data = get_history_papers(topic, query=search_query, max_results=100, date_from=date_from, date_to=date_to, min_citations=min_citations)
            history_collector.append(history_data)
            time.sleep(10)
        update_history_json(history_file, history_collector)
        logging.info(f"GET history papers end")
        return

    search_date = config.get('search_date')
    if search_date:
        try:
            datetime.datetime.strptime(search_date, '%Y%m%d')
            date_from = search_date
            date_to = search_date
            logging.info(f"Using specified date: {date_from}")
        except ValueError:
            logging.error(f"Invalid date format: {search_date}, using yesterday's date")
            yesterday = datetime.date.today() - datetime.timedelta(days=1)
            date_from = yesterday.strftime('%Y%m%d')
            date_to = yesterday.strftime('%Y%m%d')
    else:
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        date_from = yesterday.strftime('%Y%m%d')
        date_to = yesterday.strftime('%Y%m%d')
    logging.info(f"Searching papers from {date_from} to {date_to}")

    logging.info(f"GET daily papers begin")
    for topic, keyword in keywords.items():
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
        time.sleep(5)
    logging.info(f"GET daily papers end")

    if publish_readme:
        json_file = config['json_readme_path']
        md_file = config['md_readme_path']
        update_json_file(json_file, data_collector)
        json_to_md(json_file, md_file, task='Update Readme', show_badge=show_badge)

    if publish_gitpage:
        json_file = config['json_gitpage_path']
        md_file = config['md_gitpage_path']
        update_json_file(json_file, data_collector)
        json_to_md(json_file, md_file, task='Update GitPage', to_web=True, show_badge=show_badge, use_tc=False, use_b2t=False)

    if publish_wechat:
        json_file = config['json_wechat_path']
        md_file = config['md_wechat_path']
        update_json_file(json_file, data_collector_web)
        json_to_md(json_file, md_file, task='Update Wechat', to_web=False, use_title=False, show_badge=show_badge)

    feishu_webhook = config.get('feishu_webhook', '')
    if feishu_webhook:
        date_now = datetime.date.today().strftime('%Y-%m-%d')
        table_data = generate_feishu_table(data_collector, date_now)
        # Check if table_data has actual paper rows (len > 1 means has data)
        if len(table_data) > 1:
            send_to_feishun(feishu_webhook, table_data)
        else:
            # No new papers today
            send_no_papers_message(feishu_webhook)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml',
                        help='configuration file path')
    parser.add_argument('--update_history', default=False,
                        action="store_true", help='whether to update history papers with high citations')
    parser.add_argument('--date', type=str, default=None,
                        help='specify search date in YYYYMMDD format (e.g., 20260305)')
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = {**config, 'update_history': args.update_history, 'search_date': args.date}
    demo(**config)
