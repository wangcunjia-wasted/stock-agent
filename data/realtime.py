import requests


def get_realtime(code="sh600519"):
    try:
        url = f"http://hq.sinajs.cn/list={code}"
        headers = {"Referer": "https://finance.sina.com.cn/"}
        resp = requests.get(url, headers=headers)
        resp.encoding = "gbk"
        data = resp.text.split('"')[1].split(',')

        # 检查是否返回了正常的格式（至少要有 10 个字段）
        if len(data) < 10:
            raise ValueError("返回数据不完整")

        return {
            "名称": data[0],
            "当前价": float(data[3]),
            "昨收": float(data[2]),
            "最高": float(data[4]),
            "最低": float(data[5]),
            "成交额": float(data[9]),
        }
    except Exception as e:
        print(f"[警告] 获取新浪实时数据失败: {e}")
        return {
            "名称": "未知",
            "当前价": 0.0,
            "昨收": 0.0,
            "最高": 0.0,
            "最低": 0.0,
            "成交额": 0.0,
        }