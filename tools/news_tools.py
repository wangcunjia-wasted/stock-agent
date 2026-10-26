import json

import akshare as ak
import pandas as pd
import requests
from datetime import datetime, timedelta


def _fetch_news_em(pure_code: str) -> pd.DataFrame:
    """
    本地版东方财富个股新闻（akshare stock_news_em 的修复复刻）。

    为什么不复用 ak.stock_news_em：akshare 1.19.1 末尾有一行
      str.replace(r"\\u3000", "", regex=True)
    Python re 支持 \\uXXXX，但 pyarrow 后端的 RE2 引擎不支持 \\u 转义，
    装了 pyarrow 的环境必然抛 ArrowInvalid: invalid escape sequence: \\u。
    此处复刻其请求逻辑（jsonp GET，见 akshare news/news_stock.py），
    清洗全用 regex=False 的普通替换绕开 RE2。
    若日后升级 akshare/pyarrow 修复了该行，可换回 ak.stock_news_em。
    """
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    inner_param = {
        "uid": "",
        "keyword": pure_code,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": 10,
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }
    params = {
        "cb": "jQuery_local_patch",
        "param": json.dumps(inner_param, ensure_ascii=False),  # 保留中文
        "_": str(int(datetime.now().timestamp() * 1000)),
    }
    headers = {
        "accept": "*/*",
        "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/142.0.0.0 Safari/537.36"),
        "referer": f"https://so.eastmoney.com/news/s?keyword={pure_code}",
    }
    r = requests.get(url, params=params, headers=headers, timeout=15)
    # jsonp：取第一个 ( 到最后一个 ) 之间的 JSON，不依赖回调名的具体拼写
    data_json = json.loads(r.text[r.text.index("(") + 1: r.text.rindex(")")])
    temp_df = pd.DataFrame(data_json["result"]["cmsArticleWebOld"])

    # 清洗：普通替换（regex=False），RE2 不参与
    for col in ("title", "content"):
        temp_df[col] = (temp_df[col].astype(str)
                        .str.replace("<em>", "", regex=False)
                        .str.replace("</em>", "", regex=False)
                        .str.replace("\u3000", "", regex=False)
                        .str.replace("\r\n", " ", regex=False))
    temp_df["发布时间"] = pd.to_datetime(temp_df["date"])
    temp_df["新闻标题"] = temp_df["title"]
    return temp_df.sort_values("发布时间", ascending=False)


def fetch_recent_news(code="600519", limit=5):
    pure_code = code[-6:]

    # 计算时间范围：最近 30 天
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

    news_list = []

    # 1. 优先获取第一手交易所公告（巨潮资讯）
    try:
        df_notice = ak.stock_zh_a_disclosure_report_cninfo(symbol=pure_code, start_date=start_date, end_date=end_date)
        if not df_notice.empty:
            df_notice = df_notice.head(limit)  # 取最近几条
            for _, row in df_notice.iterrows():
                # 巨潮的接口列名通常是 '公告标题' 和 '公告时间'
                title = row.get('公告标题', '无标题')
                time_str = str(row.get('公告时间', ''))
                news_list.append(f"【官方公告】[{time_str}] {title}")
    except Exception as e:
        print(f"[警告] 巨潮公告获取失败: {e}")

    # 2. 补充获取媒体新闻（东方财富，走本地修复版，见 _fetch_news_em docstring）
    try:
        df_news = _fetch_news_em(pure_code).head(3)
        for _, row in df_news.iterrows():
            time_str = row['发布时间'].strftime('%Y-%m-%d %H:%M')
            title = row['新闻标题']
            news_list.append(f"【媒体报道】[{time_str}] {title}")
    except Exception as e:
        print(f"[警告] 东方财富新闻获取失败: {e}")

    if not news_list:
        return "暂无相关公告或新闻。"

    return "\n\n".join(news_list)