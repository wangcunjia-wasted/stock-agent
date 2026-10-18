import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


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

    # 2. 补充获取媒体新闻（东方财富）
    try:
        df_news = ak.stock_news_em(symbol=pure_code)
        if not df_news.empty:
            # 过滤出最近的
            df_news['发布时间'] = pd.to_datetime(df_news['发布时间'])
            df_news = df_news.sort_values(by='发布时间', ascending=False).head(3)
            for _, row in df_news.iterrows():
                time_str = row['发布时间'].strftime('%Y-%m-%d %H:%M')
                title = row['新闻标题']
                news_list.append(f"【媒体报道】[{time_str}] {title}")
    except Exception as e:
        print(f"[警告] 东方财富新闻获取失败: {e}")

    if not news_list:
        return "暂无相关公告或新闻。"

    return "\n\n".join(news_list)