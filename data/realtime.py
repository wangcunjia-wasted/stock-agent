import requests

def get_realtime(code="sh600519"):
    url = f"http://hq.sinajs.cn/list={code}"
    headers = {"Referer": "https://finance.sina.com.cn/"}
    resp = requests.get(url, headers=headers)
    resp.encoding = "gbk"
    data = resp.text.split('"')[1].split(',')
    return {
        "名称": data[0],
        "当前价": float(data[3]),
        "昨收": float(data[2]),
        "最高": float(data[4]),
        "最低": float(data[5]),
        "成交额": float(data[9]),
    }