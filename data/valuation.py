import requests

def get_valuation(code="sh600519"):
    url = f"http://qt.gtimg.cn/q={code}"
    resp = requests.get(url)
    resp.encoding = "gbk"
    data = resp.text.split('"')[1].split('~')
    return {
        "市盈率": float(data[39]) if data[39] else None,
        "市净率": float(data[46]) if data[46] else None,
        "总市值": float(data[45]) if data[45] else None,
    }